local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

-- Reference: 
-- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
-- Fukui et al. (2016) http://arxiv.org/abs/1606.01847
function CompactBilinearPooling:__init(outputSize)
  self.outputSize = outputSize
  self:reset()
end

function CompactBilinearPooling:reset()
  self.h = torch.LongTensor()
  self.s = torch.IntTensor()
  self.v = torch.Tensor()
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
  self.v:zero()
  for i=1,2 do
    if 1 == #self.h[i]:size() then  -- no batch
      for k=1,self.h[i]:size(#self.h[i]:size()) do
        self.v[i][self.h[i][k]]:add(self.s[i][k]*self.input[i][k])
      end
    else  -- batch
      for k=1,self.h[i]:size(#self.h[i]:size()) do
        self.v[i][{{},{k}}]:add(self.s[i][k]*self.input[i][{{},{k}}])
      end
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
    self.v:resize(2, self.outputSize)
  elseif 2 == #inputSize1 then  -- batch
    local batchSize = inputSizes1[1]
    self.v:resize(2, batchSize, self.outputSize)
  else
    assert(false, '# of dimensions > 2')
  end
  self:psi()
end