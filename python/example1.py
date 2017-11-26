from MDPD import *
from copy import copy
import time

starttime = time.time()

m = 100
n = 10000
c = 2

truemodel = MDPD.StageEM()
truemodel.get_config(m, n, c, c)

W, C = model_gen.crowdsourcing_rand(m, c, (0.5, 0.55))

truemodel.get_modelgen(W, C)
truemodel.copy_CW(W, C)
truemodel.gen_Data(n)
print 'Benchmark Error: '
truemodel.predict(subset=range(truemodel.m))

print '-'*40
SEM = MDPD.StageEM()
SEM.get_config(m, n, c, c)
SEM.read(truemodel.z, label=truemodel.label)
SEM.learn(stopcrit='number of iterations', num_iter=150)
SEM = misc.align(truemodel, SEM)
print 'activeset'
SEM.predict(subset=SEM.activeset)
print 'full set'
SEM.predict(subset=range(SEM.m))
print '-'*40
SEM.refine(None)
print 'fine tune & activest'
SEM.predict(None)
print 'fine tune & full set'
SEM.predict(subset=range(SEM.m))

# SEM.save('test.p')

print 1


print time.time()-starttime