import torch
import torch.nn as nn
# import kornia.color as color
import PIL
from PIL import Image,ImageFont,ImageDraw


class Painting(nn.Module):
    def __init__(self,lines):
        super(Painting, self).__init__()
        self.lines=lines
        self.device=torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noised_and_cover) :
        sentences = ['All that glitters is not gold',
                     'To thine own self be true, and it must ',
                     'follow, as the night the day, ',
                     'thou canst not then be false to any man.',
                     'Hell is empty and all the devils are here.',
                     'Love all, trust a few, do wrong to none.',
                     'By the pricking of my thumbs, Something wicked',
                     'this way comes. Open, locks, Whoever knocks!',
                     'These violent delights have violent ends...',
                     'Good night, good night! parting is such sweet ',
                     'sorrow, That I shall say good night till it be morrow.',
                     'The lady doth protest too much, methinks.',
                     'Brevity is the soul of wit.',
                     'Glendower: I can call spirits from the vasty deep.',
                     'Hotspur: Why, so can I, or so can any man; ',
                     'But will they come when you do call for them?',
                     ]
        n_sentences = self.lines
        font = '/media/dell/PROG/HJT/SteganoGAN/steganogan/Burgues_Script.otf'
        font_size = 13
        im = noised_and_cover[0].clone()
        im = PIL.Image.fromarray(im.cpu().numpy()).convert('RGBA')
        txt = Image.new('RGBA', im.size, (255, 255, 255, 0))
        print(txt.size)
        # get a font
        fnt = ImageFont.truetype(font, font_size)

        d = ImageDraw.Draw(txt)
        for i in range(min(n_sentences, len(sentences))):
            d.text((0, 0 + i * 7.8), sentences[i], font=fnt, fill=(255, 255, 255, 255))

        out = Image.alpha_composite(im, txt)
        out = out.convert('RGB')
        noised_and_cover[0]=torch.tensor(out,device=self.device)
        return noised_and_cover
