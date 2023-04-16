from itertools import count, groupby

# SIZE = 100*1024*1024 #100 MB #
SIZE = 1024 # 1024 bytes

string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed mollis bibendum neque a gravida. Fusce finibus metus condimentum accumsan volutpat. Nunc lacus urna, consequat sit amet justo at, varius maximus lacus. Integer efficitur cursus ornare. Fusce eget pretium orci. Maecenas quis fermentum orci, a rutrum lectus. Morbi vitae venenatis eros. Aliquam commodo finibus sollicitudin. Vivamus nec orci id quam iaculis facilisis. Nam scelerisque neque nec neque consectetur, iaculis tincidunt dolor vulputate. Quisque vel mi facilisis, ornare ex sed, elementum justo. Maecenas eu ligula ut massa ullamcorper hendrerit vitae sed odio. Phasellus ac nunc ut sem fermentum mollis eget et enim. Quisque non sodales ex, ut condimentum massa. Praesent interdum augue euismod sem maximus aliquet. Vestibulum quis enim fringilla, tristique felis sed, porta magna. In ac ex est. Morbi pretium nunc odio, non auctor dolor vestibulum vel. Vivamus vitae neque tincidunt, facilisis orci a, condimentum orci. Phasellus porttitor sapien nunc, et eleifend neque facilisis eget. Nulla a neque quis eros tempor congue. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Maecenas condimentum hendrerit erat tempus maximus. Pellentesque accumsan leo lorem, malesuada semper tortor cursus eget. Ut at condimentum quam. Fusce accumsan, nibh nec feugiat porttitor, diam tellus volutpat magna, ac pulvinar neque odio ut arcu. Nulla nec auctor sapien. Donec sed ex justo. Cras in sollicitudin metus, eget pretium ante. Donec at tortor eu ante gravida venenatis a ac metus. Vivamus pharetra nunc eu congue iaculis. Morbi ac mi et lectus euismod scelerisque. Aenean et placerat mi. Vestibulum viverra imperdiet arcu eu congue. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus a odio vel felis interdum egestas et et neque. Nulla porta vitae justo at lacinia. Donec in neque auctor, vulputate metus vitae, egestas est. Sed sodales varius libero, ac posuere orci dapibus non. Nunc non ante diam. Suspendisse non cursus justo. Proin dui turpis, posuere in vestibulum in, rutrum a orci. Vivamus laoreet neque vel bibendum aliquam. In vulputate tincidunt ornare. Curabitur venenatis bibendum cursus. Curabitur ante erat, congue eu ligula vel, dapibus cursus turpis. Duis erat purus, congue sit amet hendrerit eu, rhoncus at leo. Fusce ipsum ipsum, pretium sit amet mattis eu, sodales sit amet dui. Etiam eu vehicula turpis, vitae fermentum eros. Nulla hendrerit accumsan magna, in maximus nisl facilisis eu."
sep_str = string.strip().split()
joiner_str = " "

def utf8len(s):
    return len(s.encode('utf-8')) # returns size in bytes of utf-8 encoded string


class CumulativeLengthGrouper:
    def __init__(self, joiner, maxblocksize, encoding='utf-8'):
        self.encoding = encoding
        self.joinerlen = len(joiner.encode(encoding))
        self.maxblocksize = maxblocksize
        self.groupctr = count()
        self.curgrp = next(self.groupctr)
        # Special cases initial case to cancel out treating first element
        # as requiring joiner, without requiring per call special case
        self.accumlen = -self.joinerlen

    def __call__(self, newstr):
        newbytes = newstr.encode(self.encoding)
        self.accumlen += self.joinerlen + len(newbytes)
        # If accumulated length exceeds block limit...
        if self.accumlen > self.maxblocksize:
            # Move to new group
            self.curgrp = next(self.groupctr)
            self.accumlen = len(newbytes)
        return self.curgrp


def chunkify(sep_str):
    return [joiner_str.join(grp) for _, grp in groupby(sep_str, key=CumulativeLengthGrouper(joiner_str, SIZE))]


myblocks = chunkify(sep_str)

for i in myblocks:
    print(utf8len(i))
