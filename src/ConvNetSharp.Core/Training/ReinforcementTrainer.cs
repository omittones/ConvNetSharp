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
        private readonly Random rnd;

        public ReinforcementTrainer(
            Net<T> net,
            Random rnd) : base(net)
        {
            this.lossLayer = net.Layers
                .OfType<IReinforcementLayer<T>>()
                .FirstOrDefault();

            this.rnd = rnd;
        }

        public int[] Act(Volume<T> inputs)
        {
            var output = this.Net.Forward(inputs, false);
            var classCount = output.Shape.GetDimension(2);

            var values = new T[classCount];
            var prediction = new int[this.BatchSize];
            for (var n = 0; n < this.BatchSize; n++)
            {
                values[0] = output.Get(0, 0, 0, n);
                for (var j = 1; j < classCount; j++)
                    values[j] = Ops<T>.Add(values[j - 1], output.Get(0, 0, j, n));
                for (var j = 0; j < classCount; j++)
                    values[j] = Ops<T>.Divide(values[j], values[classCount - 1]);
                
                var random = Ops<T>.Cast(rnd.NextDouble());
                for (var j = 0; j < classCount; j++)
                    if (Ops<T>.GreaterThan(values[j], random))
                    {
                        prediction[n] = j;
                        break;
                    }
            }

            return prediction;
        }

        public void Reinforce(Volume<T> inputs, int[] selectedActions, T[] loss)
        {
            var outputs = this.Forward(inputs);

            this.lossLayer.SetLoss(selectedActions, loss);

            this.Backward(outputs);

            var batchSize = inputs.Shape.GetDimension(3);
            var chrono = Stopwatch.StartNew();

            //foreach (var prmtrs in this.Net.GetParametersAndGradients())
            //{
            //    prmtrs.Gradient.DoMultiply(prmtrs.Gradient, this.LearningRate);
            //    prmtrs.Gradient.DoAdd(prmtrs.Volume, prmtrs.Volume);
            //    prmtrs.Gradient.Clear();
            //}

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / batchSize;
        }
    }
}
