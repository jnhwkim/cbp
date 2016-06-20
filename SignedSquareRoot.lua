local SignedSquareRoot, parent = torch.class('nn.SignedSquareRoot', 'nn.Module')

function SignedSquareRoot:__init(args)
   parent.__init(self)
   self.module = nn.Sequential()
      :add(nn.Abs())
      :add(nn.Sqrt())
end

function SignedSquareRoot:updateOutput(input)
   self.output = self.module:forward(input)
   self.tmp = self.tmp or input.new()
   self.tmp:resizeAs(input)
   torch.sign(self.tmp, input)
   self.output:cmul(self.tmp)
   return self.output
end

function SignedSquareRoot:updateGradInput(input, gradOutput)
   local eps = 1e-1  -- to avoid gradient explosion
   torch.cmul(self.gradInput, gradOutput, 
      torch.pow(self.module:forward(input)+eps,-1)/2)
   return self.gradInput
end