using System.Linq;
using System;

namespace ConvNetSharp.Core.Training
{
    public class VanillaPolicyGradientTrainer : PolicyGradientBaseTrainer
    {
        public VanillaPolicyGradientTrainer(Net<double> net) : base(net)
        {
        }

        protected override double[] GetGradientMultipliers(Path[] paths)
        {
            var batchSize = paths.Sum(p => p.Count);

            var multipliers = new double[batchSize];
            int currentBatch = 0;
            foreach (var path in paths)
            {
                int startOfBatch = currentBatch;
                foreach (var action in path)
                {
                    multipliers[currentBatch] = action.Reward;
                    currentBatch++;
                }

                //implement reward to GO (reward_t = reward_t + ... + reward_end)
                for (var batch = currentBatch - 2; batch >= startOfBatch; batch--)
                    multipliers[batch] += multipliers[batch + 1];
            }

            //apply baseline
            var baseline = multipliers.Average();
            for (var i = 0; i < batchSize; i++)
                multipliers[i] = multipliers[i] - baseline;

            //normalize
            var stdDev = 0.0;
            for (var i = 0; i < batchSize; i++)
            {
                var r = multipliers[i];
                stdDev += r * r / multipliers.Length;
            }
            stdDev = Math.Sqrt(stdDev);
            for (var i = 0; i < batchSize; i++)
            {
                if (stdDev == 0)
                {
                    multipliers[i] = 1;
                }
                else
                {
                    var r = multipliers[i];
                    multipliers[i] = r / stdDev;
                }
            }

            this.EstimatedRewards = baseline;

            return multipliers;
        }
    }
}
