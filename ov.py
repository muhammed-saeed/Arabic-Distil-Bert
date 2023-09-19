"""
The script partition the text in Pidgin with two sets of words

Criterion for matching
 1. character:
 2. subword  : This provide more match cases for varation creation. 

most "accurate" variation.

- generated variation will be skiped if it's same as English word. For instance, `light` -> `lite` or `range` -> `rang`.
- voiceless sounds: p|t|k|f|θ|s|ʃ|h|ʧ is considered for  
"""
import re
import os
import sys
import argparse
sys.path.insert(0, os.getcwd())

from utils import read_file

### dependencies for PWLD ##
import pickle
import random
import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.spatial import distance
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

from tools.PWLD.utils import read_pickle, read_tsv, save_pickle
from collections import defaultdict, Counter
from tools.PWLD.pwld import default_dict, prepare_phone2dist, get_phonemes, compute_distance

from collections import Counter
### dependencies for PWLD ####

en_subword_to_variation_dict={
        ("k", "c"): ["k", "start", ["cc", "cy", "ci", "ce"], ["ch","ck"]], # kharacter
        ("ɔ", "a"): ["o", "middle", [], ["al"]],
        ("ɪ", "y"): ["i", "end", ["#RE#y[aeiou]"], ["y"]],
        ("ɪ", "e"): ["i", "middle", [], ["ea", "ee"]],

        # ("aʊ", "ou"): ["a", "start", [], [], []], ## TO BE REMOVED ##
        # ("ʊ",  "ou"): ["o",  "no", [], [], []],   # constrain position do not get variaiton. ## TO BE REMOVED ##
        ("ʌ", "au"): ["o", "middle", [], [], []],
        ("i", "ee"): ["i", "no", [], []],
        ("i", "ea"): ["i", "middle", ["eation"], ["ea", "ee"]],
        ("i", "eo"): ["i", "middle", [], ["ea", "ee"]],
        ("ð", "th"): ["d", "start", [], []],
        ("θ", "th"): ["t","no", [], []],      # `thing` example. but rule can be for everthing,  
        ("ŋ", "ng"): ["n", "end", [], []],
        ("f", "ph"): ["f", "middle", [], []],
        ("w", "wh"): ["w", "start", [], []],
        ("tʃ", "ch"): ["sh", "end", [], []],

        ("bəl", "ble"): ["bol", "end", [], []],
        ("pəl", "ple"): ["pol", "end", [], []],
        ("ɚ", "er"): ["a", "end", [], []],
        ("aɪt", "ight"): ["ite", "end", [], []],

        # ("i", "he"): ["e", "start", [], []],    ## TO BE REMOVED ##
        ("fɹ", "ffe"): ["f", "middle", [], []],  # e /ə/ and voice concinant will be removed 
        ("",    "e"): ["", "end", ["#RE#[aeiou][csg]e$", "#RE#[a-z]+ee", "#RE#[a-z]+lie"], []]
}

deletion_rules = {r"^(?<=[b-df-hj-np-tv-z]{1})e": ["i", "", "start"], # restric  
        r"(?:\w+)([b-df-hj-npa-tv-z]{1})e": ["ə" ,"", "middle"], # restrict `e` not in second position
}

voiceless_pattern = r"[p|t|k|f|th|s|sh|h|ch]"
# [ {}, {}, {}, {}] has len of rules
packets = [ dict() for _ in range(len(en_subword_to_variation_dict))]


# ground truth alignment
word2truePhone = {"by": [['b', 'b'], ['ɪ', 'y']],
                  "bye":[['b', 'b'], ['ɪ', 'y'], ["", 'e']],
                  "eye":[['a', 'e'], ['ɪ', 'y'], ['', 'e']]}

# Pidgin words
pidgin_word = {"dey", "wey", "sey"}


# English words
en_word = {"", "to", "go", "or", "our", "tree",
           "got", "get", "gone", "win", "won", "rang", "pin"
           "slope", "on", "h", "eat", "by", "hop", "cap"
           "brite", "lite", "us", "son", "worn",
           "ship", "past", "were", "bit", "pop",
           "band", "red", "bird", "mining", "low", "row",
           "now", "worm", "fader", "tide",
           "set", "tank", "bat", "oat", "bot"}

def contain_pattern(pattern, word):
    """Check if the string contain the pattern."""
    pattern = pattern.replace("#RE#", "")
    if re.search(pattern, word): return True
    else: return False

def merge_character(ipa2char, alignment):
    """
    Merge character into subwords given the alignment.

    Args:
      ipa2char: in the format  c a l l ||| k ɔ l
      aligment: 1-1 2-2 3-3

    Example:
        phoneme-subword pairs in a list
        [ [k, c], [u, a], [l ,l]] 

    Return:
      word (str): origin word
      paris (List): List of char and ipa pair
    """
    # Split into list of elements
    phone_seq = ipa2char.split("|||")[0].strip().split()
    char_seq = ipa2char.split("|||")[-1].strip().split()
    align_seq = alignment.split()
    
    word = "".join(char_seq)
    # select max length according to `phone_seq`
    pairs = [["",""] for _ in range(len(word))]
    # iterate the i-j alignment (phone-char)
    # access i from `phone_seq`
    # access j from `char_seq`
    # print(word)
    # print(align_seq)
    visit_char_idx = list()
    for align_pair in align_seq:
        i, j = align_pair.split("-")
        i, j = int(i), int(j)
        #print(align_pair)
        # Merge character if same phone 
        #print("phone set", phone_seq)
        #print("i", i)
        phone = phone_seq[i]
        char = char_seq[j]
        #
        visit_char_idx.append(j)

        # postprocessing
        if phone == "ɪ" and char == "e" and (j+1) == len(char_seq):
            pairs[j][0] = phone_seq[i]
            pairs[j][1] = pairs[i-1][1]
        else: 
            pairs[j][0] = phone_seq[i] 
            pairs[j][1] = char_seq[j]
            # use `=` instead `+=`
        
    # To garantee all charcters are included
    for idx in range(len(word)):
        if idx in visit_char_idx:
            continue
        pairs[idx][1] = char_seq[idx] 

    # List of pairs contain [`phone`, `subowrds`]
    # pairs = [p for p in pairs if p != ["",""]]
    word_based_on_alignment = "".join([p[1] for p in pairs])
    # if len(word) != len(word_based_on_alignment):
    #     print(word)
    #     print(pairs)
    # assert len(word) == len(word_based_on_alignment)
    return word, pairs 


def add_word_and_variation(word, phonemes, variation, word2variation):
    """
    Update the mapping 
    """
    k = (word, phonemes)
    if k not in word2variation:
        word2variation[k] = [variation]
    else:
        word2variation[k].append(variation)
    return word2variation


def write_file(fname, mapping):
    """
    Write txt file that has word, variation in each line.

    Args:
     mapping (dict): word as key and :
    """
    lst = list()
    for k, v in mapping.items():
        if type(k) == tuple:
            k = f"{k[0]}\t{k[1]} "
        lst.append(f"{k}\t{v}")

    sorted_lst = sorted(lst, key=lambda ele: (len(eval(ele.split("\t")[-1])), min(eval(ele.split("\t")[-1]))))

    with open(fname, "w") as wf:
        for line in sorted_lst:
            wf.write(f"{line}\n")
    # print(f"Writing word-variation pair to file {fname}")


def create_sample_text(word2variation, position):
    """Not use for now.

    word2variation: dict that includes all of the possible.
                    mapping that word as key and list of variations as values
    """
    test = """Na empty words dem dey tok kon dey give lie-lie promise and naw, just as dem turn field wey betta plant suppose grow to where poizin grass full, na so too dem don turn betta judgement to bad one.
      Den e sleep again kon dream anoda dream: E si seven korn head dey grow for one korn stik and dem dey fine and big well-well.
      “Salt good, but if e nor get taste again, how pesin go fit take make am sweet? Make una get salt inside una and make una dey live for peace with each oda.”Jesus komot for dat place go Judea and Jordan River aria. Plenty pipol still gada kom meet am. E      kon bigin tish dem as e dey do evritime.
      (E tok dis tin about di Spirit wey di pipol wey bilive am go get, bikos by dis time, God neva give en Spirit to pipol, bikos E neva honor Jesus Christ.)
      Stephanie continue to dey live simple life , e no spend money anyhow . This one help am fit save money wey e use as e go another country go preach .
      ( 1 Cor . 11 : 2 ) But when dem no do well , e sofri correct dem . E no hide anything from dem . <200b> — 1 Cor .
      But God take mi from di shepad work wey I dey do kon sey make I go profesai to en pipol, Israel.
      Afta una don do like dat, den God go sey, “Make una kom make wi setol di matter naw. Even doh una don stain unasef with blood bikos of una sin, I go wosh una white like snow. Even doh una stain red well-well, una go kon dey white like wool.
      Or you fit give dem work when dem come visit their country ? <200b> — Acts 18 : 1 - 3 .
      Evritime wey I remember yu for my prayers, I dey tank my God,""".strip().lower()

    import random
    seq_list = list()
    w2c = dict()
    for line in test.split("\n"):
        for tok in line.split():
            tok = tok.strip()
            if tok in word2variation:
                if tok not in w2c:
                    w2c[tok] = word2variation[tok] # multiple variation
                w = random.sample(word2variation[tok],1)[0]
                # seq_list.append(f"[{tok} -> {w}] ")

                seq_list.append(f" ({tok} -> {word2variation[tok]}) ")
            else:
                seq_list.append(tok+' ')
        seq_list.append("\n")


    o ="outputs/sample_pos.txt" if position == True else "outputs/sample_no_pos.txt"
    with open(o, "w") as wf:
        txt = "".join(seq_list)
        wf.write(txt)
    return txt

def get_sounds_from_mapping(word2variation):
    """
    word2variation: dict:

       ('easter', 'istɚ'): "['easta']\t['iːstə']\t[2]\t['0.61']"
    """
    cnt = Counter()
    for k, v in word2variation.items():
        _, sounds = k
        
        list_of_sounds = [ list(s) for s in eval(v.split("\t")[1])]
        list_of_sounds = [item for sublist in list_of_sounds for item in sublist]
        cnt.update(sounds)
        cnt.update(list_of_sounds)        
    return cnt


def update_phone2dist_with_csv(phone2dist, update_csv_file):
    # Read the csv file
    with open(update_csv_file, "r") as f:
        update_csv = f.readlines()

    for line in update_csv:
        sound_from, sound_to, ori_dist, revisit_dist, *rest = line.split(",")
        ori_dist     = float(ori_dist)
        revisit_dist = float(revisit_dist)
        # old distance
        d_ = phone2dist[sound_from][sound_to]
        
        # update the dict
        phone2dist[sound_from][sound_to] = revisit_dist
        # print(f"{sound_from} -> {sound_to}")
        # print(f"d1: {ori_dist}, to_d: {revisit_dist}")
        # print(f"d2: {d_}, to_d: {revisit_dist}")
        # print()

    # re-check. Can remove in fueature
    for line in update_csv:
        sound_from, sound_to, ori_dist, revisit_dist, *rest = line.split(",")
        assert phone2dist[sound_from][sound_to] == float(revisit_dist)
    return phone2dist


def main():
    print(read_file)
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="en.ipa2char")
    parser.add_argument("--alignment", type=str, default="tools/giza/GizaAlignments.align")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--add_distance_metric", action="store_true", default="Wether to add LD and PWLD to output files.")
    parser.add_argument("--word2ipa_path", type=str, default="tools/PWLD/data/de.tsv", help="path to the word2ipa TSV file")
    parser.add_argument("--phone2vec_path", type=str, default="tools/PWLD/data/feature_dict.p", help="path to the phone2vec pickle file")
    parser.add_argument("--pickleDirectory", type=str, default="tools/PWLD/data/phone2dist.p", help="path to phone2distance pickle file")

    ### For MixupAug ###
    # will sample k examples to augmented with new variation.
    # file will be named with `k-SIZE_sample-sample`
    parser.add_argument("--data_resource", type=str, default="data/jw300/val/val.pcm")    
    parser.add_argument("--k_size", type=int, default=100)   
    # only support `random`, and `top_k`. If use `top_k`, will use 
    # sample from top_k` candidates. We use small k 
    parser.add_argument("--sample_method", default="ramdom")   
    ### For MixupAug ###


    # revist distance featuer by csv
    args = parser.parse_args()
    print(args)
    inp_data = read_file(args.input_file).split("\n")
    align_data = read_file(args.alignment).split("\n")

    ### Prepare ###
    if args.add_distance_metric:
        word2ipa = read_tsv(args.word2ipa_path)
        phone2vec = read_pickle(args.phone2vec_path)

        PHONEME_SET = set(list(phone2vec.keys()))
        print("number of phonemes:", len(PHONEME_SET))
        print("Dimension of vector", len(phone2vec["i"]))
        phone2dist = prepare_phone2dist(args, PHONEME_SET, distance, phone2vec)


    s_one = "hændkɚtʃif"
    s_two = "hændkɐtʃiːf"
    
    d = compute_distance(s_one, s_two, phone2dist, "PWLD")
    print("distance:", d)

    # write all sounds from phone2dist
    # it takes too long, 3183 * 3183 entries
    # with open("outputs/phoneme_all.table", "w") as wf:
    #     key_sort = sorted(phone2dist.keys(), key=lambda x: len(str(x)))
    #     print("n_key", len(key_sort))
    #     key_sort = [str(ele) for ele in key_sort]
    #     col_names = ",".join(key_sort)
    #     wf.write(f"{col_names}\n")
    #     for sound_from in key_sort:
    #         sound_list = ",".join([f"{phone2dist[sound_from][s_to]:.2f}" for s_to in key_sort])
    #         wf.write(f"{sound_from},{sound_list}\n")

    # test
    print(type(phone2dist))
    print(len(phone2dist.keys()))
    for d in phone2dist:
        to_s = phone2dist[d]
        print(type(to_s))
        print(len(to_s))
        break
    
    phone2dist = update_phone2dist_with_csv(phone2dist, "outputs/revisit.csv")
    d = compute_distance(s_one, s_two, phone2dist, "PWLD")
    print("after distance:", d)



    # print(pidgin_sentence_lines)  
    ### process ###
    #en_subword_to_variation_dict = { (en_subtype_to_phone[k][0], k) : [en_subtype_to_phone[k][1], en_subtype_to_phone[k][2]] for k in en_subtype_to_phone}
    #print(en_subword_to_variation_dict)
    
    # word as key.
    # value is list of pair containing phone and subword
    word2seq_dict = dict() 
    num_match = 0
    num_match_word = 0
    word_idx = 0

    word2variation = dict()

    word2variation_multiple_rule = dict()

    variation_collection = [list() for _ in range(25)]
    print(len(variation_collection))
    
    ### Create variation ###
    ###   line 368 to 670 ###
    for ipa2char, align in zip(inp_data, align_data):
        #print(ipa2char, align)
        word, pairs = merge_character(ipa2char, align) # create sound-char pairs
        word2seq_dict[word] = pairs

        if word in pidgin_word:
            continue

        # Correcting case
        if word in word2truePhone.keys():
            pairs =  word2truePhone[word]
        
        has_variation = False

        ### for each word, we collect all feautre togethers 
        ### then do substituion once.
        multiple_rules = dict()
        # iterate all rules
        for idx, p in enumerate(en_subword_to_variation_dict):
            # convert tuple to list
            # Match: subword in word and sound in pho. sequence
            subword = p[1]
            sound = p[0]

            ipa_sequence = ipa2char.split("|||")[0].replace(" ", "")
            
            match_idx = -1 # where does (sound, subword) appear
            create_variation = False
            # Match case 1 
            if len(subword) == 1 and list(p) in pairs:

                if args.position:
                    # idx of (sound, subword) appears in original word
                    match_idx = pairs.index(list(p))
                    # Consider the position
                    if en_subword_to_variation_dict[p][1] == "start" and match_idx == 0:
                        create_variation = True
                    elif en_subword_to_variation_dict[p][1] == "middle" and match_idx != 0 and match_idx != (len(pairs)-1):
                        create_variation = True
                    elif en_subword_to_variation_dict[p][1] == "end" and match_idx == (len(pairs)-1):
                        create_variation = True
                    elif en_subword_to_variation_dict[p][1] == "no":
                        create_variation = True
                else:
                    create_variation = True


                # |NEW| set create_variation to False
                if create_variation:
                    match_idx = pairs.index(list(p))
                    negative_set = en_subword_to_variation_dict[p][2]
                    for negative_subword in negative_set:
                        if negative_subword not in word:
                            continue

                        if match_idx == word.index(negative_subword):
                            create_variation = False

                # |NEW| come / marridg
                negative_set = en_subword_to_variation_dict[p][2]
                for negative_w in negative_set:
                    if "#RE#" in negative_w and contain_pattern(negative_w, word):
                        create_variation = False

                # |NEW| update subword | subword expansion
                # if `c` can be `ch` in word. We assign `ch`    
                if create_variation:
                    match_idx = pairs.index(list(p))
                    longer_subword_set = en_subword_to_variation_dict[p][3]
                    
                    # iterate all longer subowrds. e.g. "ch" or "sch"
                    
                    for longer_w in longer_subword_set:
                        # check the substring is same as longer subword
                        substring = word[match_idx:match_idx+len(longer_w)]
                        # if longer_w in word:
                        if substring == longer_w:
                            if type(subword) == list:
                                subword.append(longer_w)
                            else:
                                subword = [subword, longer_w]
                            match_idx = -1 # reset no match_idx, do subword replacement
                    
            # Match case 2: end with `e`
            elif len(subword) == 1 and len(sound) == 0:
                variation = en_subword_to_variation_dict[p][0]
                if not word.endswith("ee") and word.endswith("e"):
                    # word, new word, phone, origin, variation
                    w = word[::-1].replace("e", "", 1)[::-1]


                    # |NEW| 
                    negative_set = en_subword_to_variation_dict[p][2]    
                    
                    for negative_w in negative_set:
                        
                        if negative_w in word:
                            create_variation = False
                        if "#RE#" in negative_w and contain_pattern(negative_w, word):
                            create_variation = False

                    if w in en_word:
                        continue

                    if create_variation:
                        out = "\t".join([str(idx), word, w, p[0], subword, variation])
                        
                        num_match += 1
                        has_variation = True

                        # Add word and variation to dict
                        word2variation = add_word_and_variation(word, ipa_sequence, w, word2variation)
                        packets[idx] = add_word_and_variation(word, ipa_sequence, w, packets[idx])

                    create_variation = False
            
            # thing
            # th-theta
            elif len(subword) > 1 and subword in word and sound in ipa_sequence:
                if args.position:
                    match_idx = word.index(subword)

                    # Consider the position
                    if en_subword_to_variation_dict[p][1] == "start" and match_idx == 0:
                        create_variation = True
                    elif en_subword_to_variation_dict[p][1] == "middle" and match_idx != 0 and match_idx != (len(word)-len(subword)):
                        create_variation = True
                    elif en_subword_to_variation_dict[p][1] == "end" and match_idx == (len(word)-len(subword)):
                        create_variation = True
                    elif en_subword_to_variation_dict[p][1] == "no":
                        create_variation = True
                else:
                    create_variation = True
            
                ### same as above ###
                # |NEW| set create_variation to False
                if create_variation:
                    match_idx = word.index(subword)
                    negative_set = en_subword_to_variation_dict[p][2]

                    for negative_subword in negative_set:
                        if negative_subword not in word:
                            continue

                        if match_idx == word.index(negative_subword):
                            create_variation = False

                # |NEW| come / marridg
                negative_set = en_subword_to_variation_dict[p][2]
                for negative_w in negative_set:
                    if "#RE#" in negative_w and contain_pattern(negative_w, word):
                        create_variation = False

                # |NEW| update subword | subword expansion
                # if `c` can be `ch` in word. We assign `ch`    
                if create_variation:
                    match_idx = word.index(subword)
                    longer_subword_set = en_subword_to_variation_dict[p][3]
                    
                    # iterate all longer subowrds. e.g. "ch" or "sch"
                    for longer_w in longer_subword_set:
                        # check the substring is same as longer subword
                        substring = word[match_idx:match_idx+len(longer_w)]
                        if substring == longer_w:
                            if type(subword) == list:
                                subword.append(longer_w)
                            else:
                                subword = [subword, longer_w]
                            match_idx = -1 # reset no match_idx, do subword replacement
                ### same as above ###


            # If true, create varaition
            if create_variation:
                variation = en_subword_to_variation_dict[p][0]

                # 
                num_replacements = 0
                # create new varation
                if match_idx != -1:            
                    w = word[:match_idx] + variation + word[match_idx+len(subword):]
                    # create new word by join the character 
                    # w = "".join([ pairs[idx][1] if idx!=match_idx else variation for idx in range(len(pairs))])
                else:

                    if type(subword) != list:
                        print("not list type")
                        w = word.replace(subword, variation, 1)
                    else:
                        word_tmp = word

                        for s in sorted(subword, key=len, reverse=True):
                            if s in word_tmp:
                                word_tmp = word_tmp.replace(s, variation, 1)
                                num_replacements+=1

                        subword = "|".join(subword)
                        w = word_tmp

                if word in ["flour", "loud", "out"]: 
                    continue

                # Continue if English word
                if w in en_word:
                    continue
                elif idx == 3 and ("ea" in w or w in ["maker"]):
                    continue
                
                # hard rule: ing -> ng. 
                if idx == 12 and word.endswith("ing"):
                    continue
                # hard rule
                if idx == 11 and word.endswith("t") and args.position:
                    continue
                
                # word, new word, phone, origin, variation
                out = "\t".join([str(idx), word, w, p[0], subword, variation])
                print("out", out)
                print(word)
                print(w)
                print()
                num_match += 1
                has_variation = True
                
                if num_replacements > 1:
                    word2variation_multiple_rule = add_word_and_variation(word, ipa_sequence, w, word2variation_multiple_rule)     
                # Add word and variation to dict
                word2variation = add_word_and_variation(word, ipa_sequence, w, word2variation) 
                packets[idx] = add_word_and_variation(word, ipa_sequence, w, packets[idx])


                # index of substring
                if "|" in subword:
                    subword_lst =subword.split("|")
                    longest_w_idx = subword_lst.index(max(subword_lst))
                    longest_w = subword_lst[longest_w_idx]
                    print("word",word)
                    print("longest_w",longest_w)
                    print("subword",subword)
                    substring_start = word.index(longest_w)
                    substring_end = substring_start + len(longest_w)
                elif type(subword) == str:
                    try:
                        substring_start = pairs.index(list(p)) # len(subword)==1
                    except:
                        substring_start = word.index(subword) # len(subword)>1
                    
                    substring_end = substring_start + len(subword)
                else:
                    print("subword", subword)
                    
                    
                # multiple_rules
                if word not in multiple_rules:
                    multiple_rules[word] = [(w, subword, variation, substring_start, substring_end)]
                else:
                    multiple_rules[word].append((w, subword, variation, substring_start, substring_end))

        if has_variation:
            num_match_word+=1
            has_variation = False


            ################## multiple rules ##################
            # boolean_lst [False, False, False]
            # string plan [False, False, False]
            # print("out", multiple_rules)
            boolean_lst = [False for _ in range(len(word))]
            word_plan = [False for _ in range(len(word))]
            

            for word, v in multiple_rules.items():
                num_applied_rules = 0
                while len(v) != 0:
                    longest_k_idx = v.index(max(v, key=lambda x:x[3]+x[4]))
                    longest_k = v[longest_k_idx]

                    subword_s = longest_k[3]
                    subword_e = longest_k[4]
                    if True not in boolean_lst[subword_s:subword_e]:
                        boolean_lst[subword_s:subword_e] = [True for _ in range(subword_e-subword_s)] # assign these range `True`
                        # get variation
                        variation = longest_k[2]
                        word_plan[subword_s:subword_e] = [variation if i == 0 else "" for i in range(subword_e-subword_s) ]

                        num_applied_rules+=1
                        # print("boolean_lst", boolean_lst)
                        # print("str_plan", word_plan)
                    print(v)
                    v.pop(longest_k_idx)
            # create new varation that apllied multiple rules
            if num_applied_rules > 1:
                if word == "character":
                    print("character")
                    print("w",w)
                new_varaition = "".join([ word_plan[idx] if word_plan[idx] != False else word[idx] for idx in range(len(word_plan))])

                word2variation_multiple_rule = add_word_and_variation(word, ipa_sequence, new_varaition, word2variation_multiple_rule) 
                word2variation = add_word_and_variation(word, ipa_sequence, new_varaition, word2variation) 
                # print("new_varaition", new_varaition)
            ################## multiple rules ##################
                    
        word_idx +=1
    ### Create variation ### 

    print("word2variation", word2variation)

    w2v = {p[0]:v for p,v in word2variation.items()}

    ##################
    # data augment
    ##################
    # data resource for augmented
    with open(args.data_resource, "r") as f:
        pidgin_sentence_lines = f.readlines()

    revisit_sentence_collect = list()
    revisit_sentence_idx     = list()

    
    for idx, sentence in enumerate(pidgin_sentence_lines):
        seq_list = list()
        revisit = False
        for tok in sentence.split():
            tok = tok.strip()
            if tok in w2v:
                w = random.sample(w2v[tok],1)[0]
                # seq_list.append(f"[{tok} -> {w}] ")
                # seq_list.append(f" ####({tok} -> {w2v[tok]}) ")

                new_tok = random.choice(w2v[tok])
                seq_list.append(new_tok)
                revisit = True
            else:
                seq_list.append(tok)
        

        new_sentence = " ".join(seq_list)

        if revisit:
            revisit_sentence_idx.append(idx)
            revisit = False
            revisit_sentence_collect.append(new_sentence)

    assert len(revisit_sentence_collect) == len(revisit_sentence_idx)
    print(len(revisit_sentence_idx))
    
    # fix seed for random indices
    random.seed(42)
    rdn_indices = random.choices(range(0, len(revisit_sentence_collect)), k=args.k_size)
    print("random k sentence idx", rdn_indices)

    for idx in rdn_indices:
        # revist 
        sent_idx, new_sent = revisit_sentence_idx[idx], revisit_sentence_collect[idx]
        pidgin_sentence_lines[sent_idx] = new_sent
        print(f"new sent: ", repr(pidgin_sentence_lines[sent_idx]))

    # output mixup_file
    last_file_name = args.data_resource.split("/")[-1]

    file_name = f"mixup_k-{args.k_size}_sample-{args.sample_method}_{last_file_name}"
    with open(file_name, "w") as wf:
        wf.write('\n'.join(pidgin_sentence_lines))

    print(f"saving mixup file to :{file_name}")
    assert 3==100


    
    
    # assert 3==2
    # print(num_match)
    # print(num_match_word)

    # add phonemes, LD and PWLD
    backend = EspeakBackend('en-us')
    total_k = 0
    for dict_idx, w2v in enumerate(packets):
        for idx, k in enumerate(w2v):
            # original word's phonemes
            word, anchor = k[0], k[1] 
            # List of variation
            variations = w2v[k]
            # List of phonemes
            phonemes = [ v.strip() for v in backend.phonemize(variations)]
            # compute distance
            ld_distance = [compute_distance(anchor, v, phone2dist, "LD") for v in phonemes]
            pwld_distance = [format(compute_distance(anchor, v, phone2dist, "PWLD"), ".2f") for v in phonemes]
            # update w2v
            w2v[k] = f"{variations}\t{phonemes}\t{ld_distance}\t{pwld_distance}"

        # print("updated w2v", w2v)
        total_k += len(w2v)
        # update new dict
        packets[dict_idx] = w2v
    
    print("w2v", w2v)
    print("key", len(w2v.keys()))
    print("created sample", create_sample_text(w2v, position=args.position))

    print("all", total_k)

    assert 3==2
    
    # write packet to each files
    for idx, d in enumerate(packets):
        # print(f"rule {idx}")
        if args.position:
            fname = f"outputs/pos/variation_rule-{idx}.txt"
        else:
            fname = f"outputs/no_pos/variation_rule-{idx}.txt"
        write_file(fname, d)

    
    for idx in range(4):
        l = packets[idx]
    backend = EspeakBackend('en-us')


    # To add phonemes of variation, we add phonemes and distance here
    num_variation = sum([len(v) for _,v in word2variation.items() ])
    print("Number of new variations", num_variation)
    for idx, k in enumerate(word2variation):
        # original word's phonemes
        word, anchor = k[0], k[1] 
        # List of variation
        variations = word2variation[k]
        # List of phonemes
        phonemes = [ v.strip() for v in backend.phonemize(variations)]
        # compute distance
        ld_distance = [compute_distance(anchor, v, phone2dist, "LD") for v in phonemes]
        pwld_distance = [format(compute_distance(anchor, v, phone2dist, "PWLD"), ".2f") for v in phonemes]
        # update word2variation
        word2variation[k] = f"{variations}\t{phonemes}\t{ld_distance}\t{pwld_distance}"
    
    # write file
    # word-phonemes -> list of variation
    o ="outputs/all_variation_pos.txt" if args.position == True else "outputs/all_variation_no_pos.txt"
    write_file(o, word2variation)

    # write phoneme vocab
    o = "outputs/phoneme_pos.vocab" if args.position == True else "outputs/phoneme_no_pos.vocab"
    sound_vocab = get_sounds_from_mapping(word2variation)
    print("word2variation", get_sounds_from_mapping(word2variation))
    with open(o, "w") as wf:
        for s, freq in sound_vocab.items():
            wf.write(f"{s},{freq}\n")

    #write_file(o, word2variation)

    ### write table ###
    reference = open(o, "r")
    reference_lst = [ line.split(",")[0] for line in reference.readlines()]
    with open(o.replace(".vocab", ".table"), "w") as wf:
        wf.write(",".join(reference_lst)+"\n")
        for sound in reference_lst:
            sound_list = ",".join([ f"{phone2dist[sound][s]:.2f}" for s in reference_lst])
            wf.write(f"{sound},{sound_list}\n")
    reference.close()

    s = sum([ phone2dist["ɜ"][v] for v in phone2dist["ɜ"]])
    ### write table
    

    ### repeat 610 to 629 for `word2variation_multiple_rule` ###
    
    # To add phonemes of variation, we add phonemes and distance here
    num_variation = sum([len(v) for _,v in word2variation_multiple_rule.items() ])
    print("Number of new variations", num_variation)
    for idx, k in enumerate(word2variation_multiple_rule):
        # original word's phonemes
        word, anchor = k[0], k[1] 
        # List of variation
        variations = word2variation_multiple_rule[k]
        # List of phonemes
        phonemes = [ v.strip() for v in backend.phonemize(variations)]
        # compute distance
        ld_distance = [compute_distance(anchor, v, phone2dist, "LD") for v in phonemes]
        pwld_distance = [format(compute_distance(anchor, v, phone2dist, "PWLD"), ".2f") for v in phonemes]
        # update word2variation_multiple_rule
        word2variation_multiple_rule[k] = f"{variations}\t{phonemes}\t{ld_distance}\t{pwld_distance}"
    
    # write file
    # word-phonemes -> list of variation
    o ="outputs/multi_rule_variation_pos.txt" if args.position == True else "outputs/multi_rule_variation_no_pos.txt"
    write_file(o, word2variation_multiple_rule)













if __name__ == "__main__":
    main()
