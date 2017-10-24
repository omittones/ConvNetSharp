using System;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class ReinforcementSoftmaxLayer<T> : SoftmaxLayer<T>, IReinforcementLayer<T>
        where T : struct, IEquatable<T>, IFormattable
    {
        private Volume<T> maxVolume;
        private T[] losses;

        public ReinforcementSoftmaxLayer(int classCount) : base(classCount)
        {
        }

        public void SetLoss(T[] losses)
        {
            this.losses = losses;
            var count = this.InputActivationGradients.Shape.GetDimension(3);
            if (count != losses.Length)
                throw new NotSupportedException("Output vs loss does not match!");

            if (this.maxVolume == null ||
               this.maxVolume.Shape.GetDimension(3) != count)
            {
                this.maxVolume = BuilderInstance<T>.Volume.SameAs(
                    this.InputActivationGradients.Storage,
                    Shape.From(1, 1, 1, count));
            }
        }

        public override void Backward(Volume<T> y, out T loss)
        {
            loss = Ops<T>.Zero;
            foreach (var item in this.losses)
                loss = Ops<T>.Add(loss, item);

            this.InputActivationGradients.MapInplace(e => Ops<T>.Zero);
            this.OutputActivation.DoMax(this.maxVolume);

            var shape = this.OutputActivation.Shape;
            var count = shape.GetDimension(3);
            for (var n = 0; n < shape.GetDimension(3); n++)
            {
                var punish = Ops<T>.GreaterThan(losses[n], Ops<T>.Zero);
                var max = this.maxVolume.Get(0, 0, 0, n);
                for (var x = 0; x < ClassCount; x++)
                {
                    var classProbability = this.OutputActivation.Get(0, 0, x, n);
                    var isHighestClass = classProbability.Equals(max);
                    if (isHighestClass)
                    {
                        this.InputActivationGradients.Set(0, 0, x, Ops<T>.Zero);
                    }
                    else
                    {
                        var amount = losses[n];
                        var gradient = Ops<T>.Cast(-0.1);
                        gradient = Ops<T>.Multiply(gradient, amount);

                        this.InputActivationGradients.Set(0, 0, x, gradient);
                    }
                }
            }
        }
    }
}
