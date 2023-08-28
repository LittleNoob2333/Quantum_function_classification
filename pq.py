import deepquantum as dq

cir = dq.QubitCircuit(4)
cir.hlayer()
cir.rxlayer([0,2])
cir.barrier()
cir.rylayer([1,3])
cir.barrier()
cir.u3layer()
cir.barrier()
cir.cxlayer()
cir.barrier()
cir.any(ora)
cir.draw(filename ='p.png')