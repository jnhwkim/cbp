require 'nn'
require 'cunn'

local cbptest = torch.TestSuite()
local precision = 1e-5
local debug = false

function cbptest.testPsi()
   local homogeneous = true
   local batch = 1
   local dim = 5
   local S = 100
   local x = torch.rand(batch,dim):cuda()
   local y = torch.rand(batch,dim):cuda()
   local tmp = 0
   local c
   for s=1,S do
      c = nn.CompactBilinearPooling(dim, homogeneous):cuda()
      c:forward({x,y})
      tmp = tmp + c.y[1][1]:dot(c.y[2][1])
   end
   local xy = x[1]:dot(y[1])
   local diff = math.abs(tmp/S - xy)
   assert(diff / xy < .1, 'error_ratio='..diff / xy..', E[<phi(x,h,s),phi(y,h,s)>]=<x,y>')
end

function cbptest.testConv()
   local x = torch.CudaTensor{{1,2,3},{2,3,4}}
   local y = torch.CudaTensor{{1,1,1},{2,2,2}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{6,6,6},{18,18,18}}
   local output = torch.CudaTensor()
   output = c:conv(x,y)  -- cuda only
   ans:add(-output)
   assert(ans:norm() < precision, ans:norm())

   local x = torch.CudaTensor{{1,2,3,1,1},{2,3,4,1,1}}
   local y = torch.CudaTensor{{1,1,1,1,1},{2,2,2,1,1}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{8,8,8,8,8},{17.5,17.5,17.75,17.75,17.5}}
   output = c:conv(x,y)
   ans:add(-output)
   assert(ans:norm() < precision, ans:norm())
end

function cbptest.testLearning()
   cutorch.manualSeed(12)
   local N = 1000
   local M = 2
   local C = 2
   local iter = 20
   local lr = .05
   local x = torch.rand(N,M)
   local y = torch.rand(N,M)
   local t = torch.cmul(x[{{},{2}}],y[{{},{2}}]) + torch.cmul(x[{{},{1}}],y[{{},{1}}])

   x=x:cuda()
   y=y:cuda()
   t=t:cuda()

   local c = nn.CompactBilinearPooling(C)
   local model = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Sequential()
            :add(nn.Linear(M,M))
            :add(nn.Tanh()))
         :add(nn.Sequential()
            :add(nn.Linear(M,M))
            :add(nn.Tanh())))
      :add(c)
      :add(nn.SignedSquareRoot())
      :add(nn.Normalize(2))
      :add(nn.Linear(C,1))

   -- baseline
   -- local model = nn.Sequential():add(nn.JoinTable(2)):add(nn.Linear(C*2,1))
   
   local criterion = nn.MSECriterion()

   model=model:cuda()
   criterion=criterion:cuda()

   model:getParameters():uniform(-.08,.08)

   if debug then print(' ') end
   for i=1,iter do
      local output = model:forward{x,y}
      J = criterion:forward(output, t)
      if 0==i%1 and debug then print(J) end
      local dt = criterion:backward(output, t)
      model:zeroGradParameters()
      model:backward({x,y},dt)
      model:updateParameters(lr)

      if i==1 and debug then
         print(c.h)
         print(c.s)
      end
   end
   assert(J < .2, 'CBP failed to learn (J='..J..')')
end

function cbp.test(tests, _debug)
   debug = _debug or false
   mytester = torch.Tester()
   mytester:add(cbptest)
   math.randomseed(os.time())
   mytester:run(tests)
end