# regex
# %%
import re

text_to_search = '''
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890

Ha HaHa

MetaCharacters (Need to be escaped!):
. ^ $ * + ? { } [ ] \ | ( )

coreyms.com

321-555-4321
123.555.1234

Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr.  T
'''

sentence = 'Start a sentence and then bring it to an end'

# %%
pattern = re.compile(r'abc')

matches = pattern.finditer(text_to_search)

for match in matches:
    print(match)
    
print(text_to_search[1:4])

# %%
pattern = re.compile(r'_\._')

pattern = re.compile(r'coreyms\.com')

pattern = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')

# %%
with open('data.txt', 'r', encoding = 'utf-8') as f:
    contents = f.read()
    
    matches = pattern.finditer(text_to_search)
    
    for match in matches:
        print(match)

# %%
pattern = re.compile(r'[89]00[-.]\d[A-Z]')

pattern = re.compile(r'[a-zA-Z]')

pattern = re.compile(r'[^a-zA-z]')

pattern = re.compile(r'M(r|s|rs)\.?\s[A-Z]\w+')

pattern = re.compile(r'start', re.IGNORECASE)
