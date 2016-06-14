local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

require 'cunn'

-- Reference: 
-- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
-- Fukui et al. (2016) http://arxiv.org/abs/1606.01847
function CompactBilinearPooling:__init(outputSize, homogeneous)
   self.outputSize = outputSize
   self.homogeneous = homogeneous
   self.tmp = torch.Tensor()
   self:reset()
end

function CompactBilinearPooling:reset()
   self.h = torch.LongTensor()
   self.s = torch.IntTensor()
   self.y = torch.Tensor()
   self.output = torch.Tensor()
end

function CompactBilinearPooling:sample()
   for i=1,2 do
      for k=1,self.h[i]:size(#self.h[i]:size()) do
         self.h[i][k] = torch.random(1,self.outputSize)  -- sample from (1,..,C)
         self.s[i][k] = torch.random(0,1)*2-1  -- sample from (-1,1)
      end
   end
end

function CompactBilinearPooling:psi()
   self.y:zero()
   for i=1,2 do
      if self.homogeneous then  -- using the same samples
         for j=1,self.h[1]:size(#self.h[1]:size()) do
           local y_ = self.y[i][self.h[1][j]]
           self.y[i][self.h[1][j]] = y_ + self.s[1][j]*self.input[i][j]
        end
      else
         for j=1,self.h[i]:size(#self.h[i]:size()) do
            if 1 == #self.input[i]:size() then
               local y_ = self.y[i][self.h[i][j]]
               y_ = y_ + self.s[i][j]*self.input[i][j]
            elseif 2 == #self.input[i]:size() then
               local y_ = self.y[i][{{},{self.h[i][j]}}]
               y_ = y_:add(self.s[i][j]*self.input[i][{{},{j}}])
            end
         end
      end
      self.y:typeAs(self.input[i])
   end
end

function CompactBilinearPooling:convfft(x, y)
   if not self.module then
      self.module = nn.TemporalConvolution(1, 1, x:size(#x:size()))
      self.module.weight=self.module.weight:typeAs(x)
      self.module.bias=self.module.bias:typeAs(x)
      self.module.output=self.module.output:typeAs(x)
   end
   self.module.bias:zero()
   local function calc(x,y)
      local dim = y:size(#y:size())
      self.module.weight:copy(x)
      self.tmp=self.tmp:typeAs(y)
      self.tmp:resize(1,dim*2-1,1):zero()
      self.tmp[{{},{dim,dim*2-1},{}}]:copy(y)  -- left-padding
      return self.module:forward(self.tmp)
   end
   self.output = self.output:typeAs(x)
   if 1 == #x:size() then
      self.output:resizeAs(x)
      self.output:copy(calc(x,y))
      return self.output
   elseif 2 == #x:size() then
      self.output:resizeAs(x)
      for i=1,x:size(1) do
         require 'sys'
         sys.tic()
         self.output[i]:copy(calc(x[i],y[i]))
         print(sys.toc())
      end
   end
end

function CompactBilinearPooling:updateOutput(input)
   self.input = input
   local inputSizes1 = input[1]:size()
   local inputSizes2 = input[2]:size()
   local sizes1 = inputSizes1[#inputSizes1]
   local sizes2 = inputSizes2[#inputSizes2]
   self.h:resize(2, sizes1)
   self.s:resize(2, sizes1)
   self:sample()

   if 2 > #inputSizes1 then  -- no batch
      self.y:resize(2, self.outputSize)
   elseif 2 == #inputSizes1 then  -- batch
      local batchSize = inputSizes1[1]
      self.y:resize(2, batchSize, self.outputSize)
   else
      assert(false, '# of dimensions > 2')
   end
   self.y=self.y:typeAs(input[1])
   self:psi()

   self:convfft(self.y[1], self.y[2])
end