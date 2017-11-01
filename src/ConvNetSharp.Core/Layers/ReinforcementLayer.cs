using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class ReinforcementLayer<T> : SoftmaxLayer<T>, IReinforcementLayer<T>, ILastLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        private int classCount;
        private T[] losses;
        private int[] selectedActions;

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            if (inputWidth != 1 || inputHeight != 1)
                throw new NotSupportedException();

            this.classCount = inputDepth;
        }

        public void SetLoss(int[] selectedActions, T[] losses)
        {
            this.losses = losses;
            this.selectedActions = selectedActions;
            var count = this.InputActivationGradients.Shape.GetDimension(3);
            if (count != losses.Length)
                throw new NotSupportedException("Output vs loss does not match!");
        }

        public override Volume<T> DoForward(Volume<T> input, bool isTraining = false)
        {
            return base.DoForward(input, isTraining);
        }

        public override Volume<T> Forward(bool isTraining)
        {
            return base.Forward(isTraining);
        }

        public override void Backward(Volume<T> y, out T loss)
        {
            base.Backward(y, out loss);

            var shape = this.OutputActivation.Shape;
            var batches = shape.GetDimension(3);

            loss = Ops<T>.Zero;
            this.InputActivationGradients.Clear();
            for (var batch = 0; batch < batches; batch++)
            {
                //amount = Ops<T>.Negate(amount);
                //var classProbability = this.OutputActivation.Get(0, 0, x, n);
                var amount = losses[batch];
                loss = Ops<T>.Add(loss, Ops<T>.Multiply(amount, amount));
                this.InputActivationGradients.Set(0, 0, selectedActions[batch], batch, amount);
            }
        }
    }
}
