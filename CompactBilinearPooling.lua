local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

require 'spectralnet'

-- Reference: 
-- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
-- Fukui et al. (2016) http://arxiv.org/abs/1606.01847
function CompactBilinearPooling:__init(outputSize, homogeneous)
   assert(outputSize and outputSize >= 1, 'missing outputSize!')
   self.outputSize = outputSize
   self.homogeneous = homogeneous
   self:reset()
   self.debug = false
end

function CompactBilinearPooling:reset()
   self.h1 = torch.Tensor()
   self.h2 = torch.Tensor()
   self.s1 = torch.Tensor()
   self.s2 = torch.Tensor()
   self.y = torch.Tensor()
   self.gradInput = {}
   self.tmp = torch.Tensor()
end

function CompactBilinearPooling:sample()
   self.h1:uniform(0,self.outputSize):ceil()
   self.h2:uniform(0,self.outputSize):ceil()
   self.s1:uniform(0,2):floor():mul(2):add(-1)
   self.s2:uniform(0,2):floor():mul(2):add(-1)
end


function CompactBilinearPooling:psi()
   self.y:zero()
   local batchSize = self.input[1]:size(1)
   for i=1,2 do
      if self.homogeneous then  -- using the same samples
         self.y[i]:indexAdd(2,self.h[1],torch.cmul(self.s[1]:repeatTensor(batchSize,1),self.input[i]))
      else
	 if i==1 then
           self.y[i]:indexAdd(2,self.h1,torch.cmul(self.s1:repeatTensor(batchSize,1),self.input[i]))
	 else
	   self.y[i]:indexAdd(2,self.h2,torch.cmul(self.s2:repeatTensor(batchSize,1),self.input[i]))
	 end
      end
   end
end

function CompactBilinearPooling:conv(x,y)
   local batchSize = x:size(1)
   local dim = x:size(2)
   local function makeComplex(x,y)
      self.x_ = self.x_ or torch.CudaTensor()
      self.x_:resize(x:size(1),1,1,x:size(2),2):zero()
      self.x_[{{},{1},{1},{},{1}}]:copy(x)
      self.y_ = self.y_ or torch.CudaTensor()
      self.y_:resize(y:size(1),1,1,y:size(2),2):zero()
      self.y_[{{},{1},{1},{},{1}}]:copy(y)
   end
   makeComplex(x,y)
   self.fft_x = self.fft_x or torch.CudaTensor(batchSize,1,1,dim,2)
   self.fft_y = self.fft_y or torch.CudaTensor(batchSize,1,1,dim,2)
   local output = output or torch.CudaTensor()
   output:resize(batchSize,1,1,dim*2)
   cufft.fft1d(self.x_:view(x:size(1),1,1,-1), self.fft_x)
   cufft.fft1d(self.y_:view(y:size(1),1,1,-1), self.fft_y)
   cufft.ifft1d(self.fft_x:cmul(self.fft_y), output)
   return output:resize(batchSize,1,1,dim,2):select(2,1):select(2,1):select(3,1)
end

function CompactBilinearPooling:updateOutput(input)
   if self.debug then sys.tic(1) end
   self.input = input
   local inputSizes1 = input[1]:size()
   local inputSizes2 = input[2]:size()

   if 0==#self.h1:size() then
      self.h1:resize(inputSizes1[#inputSizes1])
      self.h2:resize(inputSizes2[#inputSizes2])
      self.s1:resize(inputSizes1[#inputSizes1])
      self.s2:resize(inputSizes2[#inputSizes2]) 
      self:sample()  -- samples are fixed
   end

   if 2 > #inputSizes1 then  -- no batch
      self.y:resize(2, self.outputSize)
   elseif 2 == #inputSizes1 then  -- batch
      local batchSize = inputSizes1[1]
      self.y:resize(2, batchSize, self.outputSize)
   else
      assert(false, '# of dimensions > 2')
   end
   if self.debug then print('pre:', sys.toc(1)) end
   self:psi()
   if self.debug then print('psi:', sys.toc(1)) end
   self.output = self:conv(self.y[1], self.y[2])
   if self.debug then print('conv:', sys.toc(1)) end

   return self.output
end

function CompactBilinearPooling:updateGradInput(input, gradOutput)
   local dim = input[1]:size(2)
   local batchSize = input[1]:size(1)
   self.gradInput = self.gradInput or {}

   for k=1,2 do
      self.gradInput[k] = self.gradInput[k] or input[k].new()
      self.gradInput[k]:resizeAs(input[k]):zero()
      self.tmp = self.tmp or gradOutput.new()
      self.tmp:resizeAs(gradOutput)

      self.tmp = self:conv(gradOutput, self.y[k%2+1])
      if k==1 then
	self.gradInput[k]:index(self.tmp, 2, self.h1)
	self.gradInput[k]:cmul(self.s1:repeatTensor(batchSize,1))
      else
	self.gradInput[k]:index(self.tmp, 2, self.h2)
	self.gradInput[k]:cmul(self.s2:repeatTensor(batchSize,1))
      end
   end

   return self.gradInput
end
