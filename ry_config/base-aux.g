world { X:[0 0 .1] }

cam0(world){ Q:"t(0 0 10) d(180 1 0 0)" shape:camera width:300 height:300}

#thick radius floor and walls
base (world){ shape:ssBox, Q:[0 0 -0.1], size:[10 10 .1 .04], color:[0 0 0], contact: 0 friction:10, logical:{table} }
floor (world){ shape:ssBox, Q:[0 0 -0.05], size:[4.1 4.1 .1 .04], color:[0 255 0], contact: 1 friction:10, logical:{table} }

outwall_right (world){ shape:ssBox, Q:[0 -2. 0.2], size:[4.1 .1 0.4 .04], color:[1 0 0], contact: 1 }
outwall_back (world){ shape:ssBox, Q:[2. 0 0.2], size:[.1 4.1 0.4 .04], color:[1 0 0], contact: 1 }
outwall_left (world){ shape:ssBox, Q:[0 2. 0.2], size:[4.1 .1 0.4 .04], color:[1 0 0], contact: 1 }
outwall_front (world){ shape:ssBox, Q:[-2. 0 0.2], size:[.1 4.1 0.4 .04] , color:[1 0 0], contact: 1 }

egoJoint(world){Q:[0 0 .2]  }
ego(egoJoint) {
    shape:ssCylinder, Q:[-1.3 -1.3, 0], size:[.4 .02 .02], color:[1 1 0]
}