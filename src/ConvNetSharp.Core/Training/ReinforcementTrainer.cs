using ConvNetSharp.Volume;
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;

namespace ConvNetSharp.Core.Training
{
    public class ReinforcementTrainer<T> : SgdTrainer<T>
        where T : struct, IEquatable<T>, IFormattable

    {
        private readonly IReinforcementLayer<T> lossLayer;

        public ReinforcementTrainer(Net<T> net) : base(net)
        {
            this.lossLayer = net.Layers
                .OfType<IReinforcementLayer<T>>()
                .FirstOrDefault();
        }

        public void Reinforce(Volume<T> inputs, T[] loss)
        {
            var outputs = this.Forward(inputs);

            this.lossLayer.SetLoss(loss);

            this.Backward(outputs);

            var batchSize = inputs.Shape.GetDimension(3);
            var chrono = Stopwatch.StartNew();

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;
        }
    }
}
