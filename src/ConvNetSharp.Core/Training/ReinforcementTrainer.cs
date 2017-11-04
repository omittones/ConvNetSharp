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
        private Volume<T> inputs;

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
            this.inputs = inputs;

            var output = this.Forward(inputs);

            Debug.Assert(output.Width == 1);
            Debug.Assert(output.Height == 1);
            Debug.Assert(output.BatchSize == this.BatchSize);

            var classCount = output.Depth;
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

        public void Reinforce(int[][] pathActions, T[] rewards)
        {
            //Discount Returns r0 = (r0 + r1 * gamma + r2 * gamma^2 + r2 * gamma^3 ...)
            //Advantage = Returns - baseline

            this.lossLayer.SetLoss(pathActions, rewards);

            //refit baseline to minimize Sum[(R - b)^2]
            //var mean = averages.Average();
            //averages = averages.Select(a => a - mean).ToArray();
            //var stdev = averages.Select(a => a * a).Average();
            //averages = averages.Select(a => a / stdev).ToArray();

            this.Backward(this.inputs);

            var chrono = Stopwatch.StartNew();

            //foreach (var prmtrs in this.Net.GetParametersAndGradients())
            //{
            //    prmtrs.Gradient.DoMultiply(prmtrs.Gradient, this.LearningRate);
            //    prmtrs.Gradient.DoAdd(prmtrs.Volume, prmtrs.Volume);
            //    prmtrs.Gradient.Clear();
            //}

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / this.inputs.BatchSize;
        }
    }
}
