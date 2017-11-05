using ConvNetSharp.Volume;
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;

namespace ConvNetSharp.Core.Training
{
    public class ReinforcementTrainer : SgdTrainer<double>
    {
        public bool ApplyBaselineAndNormalizeReturns { get; set; }
        public double Baseline { get; private set; }
        public double RewardDiscountGamma { get; set; }
        public double EstimatedRewards => Ops<double>.Negate(this.Loss);

        private readonly IReinforcementLayer<double> finalLayer;
        private readonly Random rnd;

        public ReinforcementTrainer(
            Net<double> net,
            Random rnd) : base(net)
        {
            this.finalLayer = net.Layers
                .OfType<IReinforcementLayer<double>>()
                .FirstOrDefault();

            this.rnd = rnd;
            this.Baseline = Ops<double>.Zero;
            this.RewardDiscountGamma = Ops<double>.Zero;
            this.ApplyBaselineAndNormalizeReturns = true;
        }

        public int[] Act(Volume<double> inputs)
        {
            var output = this.Forward(inputs);

            Debug.Assert(output.Width == 1);
            Debug.Assert(output.Height == 1);
            Debug.Assert(output.BatchSize == this.BatchSize);

            var classCount = output.Depth;
            var values = new double[classCount];
            var prediction = new int[this.BatchSize];
            for (var n = 0; n < this.BatchSize; n++)
            {

                int max = 0;
                for (var j = 0; j < classCount; j++)
                    if (Ops<double>.GreaterThan(output.Get(0, 0, j, n), output.Get(0, 0, max, n)))
                        max = j;
                prediction[n] = max;
            }

            return prediction;
        }

        public override void Train(Volume<double> x, Volume<double> y)
        {
            throw new NotSupportedException("Use Reinforce method instead!");
        }

        public void Reinforce(Volume<double> pathInputs, int[][] pathActions, double[] pathReturns)
        {
            this.Forward(pathInputs);

            if (ApplyBaselineAndNormalizeReturns)
            {
                //normalize by standard deviation
                var stdDev = pathReturns
                    .Select(a => { var d = a - Baseline; d = d * d; return d; })
                    .Average();

                pathReturns = pathReturns
                    .Select(r => r / stdDev)
                    .ToArray();
            }

            this.finalLayer.SetReturns(pathActions, pathReturns, this.Baseline, this.RewardDiscountGamma);

            if (ApplyBaselineAndNormalizeReturns)
            {
                //refit baseline to minimize Sum[(R - b)^2]
                Baseline = Ops<double>.Zero;
                for (var i = 0; i < pathReturns.Length; i++)
                    Baseline = Ops<double>.Add(pathReturns[i], Baseline);
                Baseline = Ops<double>.Divide(Baseline, Ops<double>.Cast(pathReturns.Length));
            }

            this.Backward(pathInputs);

            var chrono = Stopwatch.StartNew();

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / pathInputs.BatchSize;
        }
    }
}
