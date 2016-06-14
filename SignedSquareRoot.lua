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
   gradOutput:cmul(self.tmp)
   self.gradInput = self.module:backward(input, gradOutput)
   return self.gradInput
end