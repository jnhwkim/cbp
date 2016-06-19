require 'nn'
require 'cunn'

local cbptest = torch.TestSuite()
local precision = 1e-5

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
   cutorch.manualSeed(123)
   local N = 100
   local C = 3
   local iter = 100
   local lr = .05
   local x = torch.rand(N,1)
   local y = torch.rand(N,1)
   local t = torch.pow(x,2) + torch.pow(y,2) + torch.cmul(x,y)

   x=x:cuda()
   y=y:cuda()
   t=t:cuda()

   local model = nn.Sequential()
      :add(nn.CompactBilinearPooling(C))
      :add(nn.SignedSquareRoot())
      :add(nn.Linear(C,1))
   local criterion = nn.MSECriterion()

   model=model:cuda()
   criterion=criterion:cuda()

   for i=1,iter do
      local output = model:forward{x,y}
      J = criterion:forward(output, t)
      local dt = criterion:backward(output, t)
      model:zeroGradParameters()
      model:backward({x,y},dt)
      model:updateParameters(lr)
   end
   assert(J < .2, 'CBP failed to learn (J='..J..')')
end

function cbp.test(tests)
   mytester = torch.Tester()
   mytester:add(cbptest)
   math.randomseed(os.time())
   mytester:run(tests)
end