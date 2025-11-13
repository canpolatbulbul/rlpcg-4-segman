world { X:[0 0 .1] }
camera_gl(world){ Q:"t(0 0 16) d(180 1 0 0)" shape:camera width:900 height:900}
floor (world){ shape:ssBox, Q:[0 0 -0.05], size:[4.1 4.1 .1 .04], color:[0.7686 0.6863 .6471], contact: 1 friction:10, logical:{table} }
#########################################################################################################################
outwall_right (world){ shape:ssBox, Q:[0 -2. 0.2], size:[4.1 .1 0.4 .04], color:[0.6953 0.515625 .453125], contact: 1 }
outwall_back (world){ shape:ssBox, Q:[2. 0 0.2], size:[.1 4.1 0.4 .04], color:[0.6953 0.515625 .453125], contact: 1 }
outwall_left (world){ shape:ssBox, Q:[0 2. 0.2], size:[4.1 .1 0.4 .04], color:[0.6953 0.515625 .453125], contact: 1 }
outwall_front (world){ shape:ssBox, Q:[-2. 0 0.2], size:[.1 4.1 0.4 .04] , color:[0.6953 0.515625 .453125], contact: 1 }
#########################################################################################################################
egoJoint(world){Q:[0 0 0.1]  }
ego(egoJoint) {
    shape:ssCylinder, Q:[-1.3 -1.3 0], size:[.2 .2 .02], color:[0.96875 0.7421875 0.30859375], logical:{agent}, limits: [-4 4 -4 4], sampleUniform:.5,
    joint:transXY, contact: 1
}




