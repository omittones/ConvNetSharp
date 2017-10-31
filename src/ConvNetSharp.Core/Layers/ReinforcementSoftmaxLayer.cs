using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class ReinforcementSoftmaxLayer<T> : SoftmaxLayer<T>, IReinforcementLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> maxes;
        private T[] losses;
        private int[] selectedActions;

        public ReinforcementSoftmaxLayer(int classCount) : base(classCount)
        {
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
            loss = Ops<T>.Zero;
            foreach (var item in this.losses)
                loss = Ops<T>.Add(loss, item);

            var shape = this.OutputActivation.Shape;
            var count = shape.GetDimension(3);
            for (var n = 0; n < count; n++)
            {
                for (var x = 0; x < ClassCount; x++)
                {
                    var amount = losses[n];
                    var classProbability = this.OutputActivation.Get(0, 0, x, n);
                    var wasPredicted = selectedActions[n] == x;
                    if (wasPredicted)
                        this.InputActivationGradients.Set(0, 0, x, amount);
                    else
                        this.InputActivationGradients.Set(0, 0, x, Ops<T>.Zero);
                }
            }
        }
    }
}
