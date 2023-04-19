# color code

'''
this code for...
original image color convert to some fixed color.
and this color hex code is define fixed color.

but converted color is strange. so this code have to change.
'''

class HexColorCode:
    def __init__(self):
        self.hexColorCode_example = """
87CEEB,
00A4B4,
4166F5,
384883,
002A77,
000F89,
003153,
8F00FF,
22313C,
6699CC,
AAA9AD,
826644,
704214,
8A3324,
664228,
B87333,
9DBCD4,
E6BE8A,
BCBABE,
035489,C5C6C8,2D1C62,FFFFF0,BCBC82,
EAEAEA,300D11,0066B1,C6862C,
2A469A,009FD9,BFC4C8,FFD49F,
4E7280,49312D,88785F,928362,
55758A,221E1F,797337,682D01,
4F3210,284080,CF912C,69747A,
CE9933,0D68B1,
E78882,EF6575,F5901E,F16A26,
F7B7AB,FEE0BE,FFD49F,FFFFFF,
933837,49312D,300D11,928362,
C5D3ED,F59251,F27920,EAEAEA,81A8D3,
FFFFFF,F5F2CF,E6B7CB,DB8CA9,
FC4F87,A84C99,CC2133,A71931,
F3D5B1,DC7E38,E09486,FBE22E,
E7A632,BE994B,DEE7AE,D7DD3F,
4F994A,79BFB5,4D9D9A,A5D1F4,
265297,1B3B64,AA92C2,5E56A1,
462864,8F7660,AA7F52,5B3101,
A4A8A9,161513
"""
        self.colorName_example = """
932sky blue,
933peacock blue,
934ultramarine light,
935ultramarine deep,
937marine blue,
938phthalocyanine blue,
939prussian blue,
941violet,
992pearl indigo,
962blue gray,
961silver gray,
929raw umber,
928sepia,
925burnt umber,
927vandyke brown,
567Copper,
996Pale gold,
966Grayish Blue,
1)french gray,
928,964,934,907,976,
929,961,995,906,
987,984,935,962,
922,999,963,925,
966,997,992,960,
905,939,955,941,
938,937,
911,916,912,913,910,909,906,901,924,928,927,963,972,979,989,987,991,
White,Antique White,Cherry Blossom Pink,Wild Rose Pink,
Tutti Fruitti,Fun Fuchsia,Bright Red,Christmas Red,
Natural Beige,Bright Orange,Bright Coral,Bright Yellow,
Goldenrod,Antique Gold,Early Spring Green,Citrus Green,
Holiday Green,Turquoise,Dark Turquoise,Cool Blue,
Bright Blue,Navy Blue,Lavender,Purple Passion,
African Violet,Country Maple,Golden Brown,Cinnamon Brown,
Storm Cloud Grey,Black
"""
        
        self.hexColorCode = """

"""
        self.colorName = """

"""

        self.hexColorCodeList = self.hexColorCode.replace("\n", "").split(",")
        self.colorNameList = self.colorName.replace("\n", "").split(",")
        
if __name__ == "__main__":
    code = HexColorCode().hexColorCodeList
    
            
        