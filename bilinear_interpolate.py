# -*- coding: utf-8 -*-
'''
Copyright (c) 2014 The MatConvNet team.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the <organization>. The name of the
<organization> may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
'''

import numpy as np

def bilinear_interpolate(im, x, y):
  x = np.asarray(x)
  y = np.asarray(y)

  x0 = np.floor(x).astype(int)
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y1 = y0 + 1

  x0 = np.clip(x0, 0, im.shape[1]-1);
  x1 = np.clip(x1, 0, im.shape[1]-1);
  y0 = np.clip(y0, 0, im.shape[0]-1);
  y1 = np.clip(y1, 0, im.shape[0]-1);

  Ia = im[ y0, x0 ]
  Ib = im[ y1, x0 ]
  Ic = im[ y0, x1 ]
  Id = im[ y1, x1 ]

  wa = (1-x+x0) * (1-y+y0)
  wb = (1-x+x0) * (y-y0)
  wc = (x-x0) * (1-y+y0)
  wd = (x-x0) * (y-y0)

  wa = wa.reshape(x.shape[0], x.shape[1], 1)
  wb = wb.reshape(x.shape[0], x.shape[1], 1)
  wc = wc.reshape(x.shape[0], x.shape[1], 1)
  wd = wd.reshape(x.shape[0], x.shape[1], 1)

  return wa*Ia + wb*Ib + wc*Ic + wd*Id