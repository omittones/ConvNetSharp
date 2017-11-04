using System;
using ConvNetSharp.Volume;
using System.Diagnostics;
using System.Linq;
using Vol = ConvNetSharp.Volume.Volume<double>;

namespace ConvNetSharp.Core.Layers
{
    public class ReinforcementLayer : LastLayerBase<double>, IReinforcementLayer<double>, ILastLayer<double>
    {
        private VolumeBuilder<double> build = BuilderInstance<double>.Volume;

        private int classCount;
        private double[] rewards;
        private int[][] pathActions;

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            if (inputWidth != 1 || inputHeight != 1)
                throw new NotSupportedException("Unsupported input format!");

            this.classCount = inputDepth;
            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
            this.OutputDepth = inputDepth;

            base.Init(inputWidth, inputHeight, inputDepth);
        }

        public void MuPolicy(Vol input, Vol output)
        {
            var maxes = build.SameAs(1, 1, 1, input.BatchSize);
            input.DoMax(maxes);
            for (var bs = 0; bs < input.BatchSize; bs++)
            {
                double expSum = 0;
                for (var d = 0; d < input.Depth; d++)
                {
                    var exp = Math.Exp(input.Get(0, 0, d, bs) - maxes.Get(0, 0, 0, bs));
                    expSum += exp;
                    output.Set(0, 0, d, bs, exp);
                }

                for (var d = 0; d < input.Depth; d++)
                {
                    var exp = output.Get(0, 0, d, bs);
                    output.Set(0, 0, d, bs, exp / expSum);
                }
            }
        }

        public void LogMuPolicy(Vol input, Vol output)
        {
            var maxes = build.SameAs(1, 1, 1, input.BatchSize);
            input.DoMax(maxes);
            for (var bs = 0; bs < input.BatchSize; bs++)
            {
                var max = maxes.Get(0, 0, 0, bs);

                double expSum = 0;
                for (var d = 0; d < input.Depth; d++)
                    expSum += Math.Exp(input.Get(0, 0, d, bs) - max);
                var logSumExp = max + Math.Log(expSum);

                for (var d = 0; d < input.Depth; d++)
                {
                    var value = input.Get(0, 0, d, bs);
                    output.Set(0, 0, d, bs, value - logSumExp);
                }
            }
        }

        public void GradientLogMuPolicy(Vol input, Vol mu, int action, int batch, Vol output)
        {
            Debug.Assert(input.Depth == output.Depth);
            Debug.Assert(input.BatchSize == output.BatchSize);

            for (var d = 0; d < input.Depth; d++)
            {
                var flag = d == action ? 1.0 : 0.0;
                output.Set(0, 0, d, batch, flag - mu.Get(0, 0, d, batch));
            }
        }

        public void PathLikelihoodRatio(Vol input, Vol mu, int[] actions, int startBatchIndex, Vol output)
        {
            Debug.Assert(input.Depth == output.Depth);
            Debug.Assert(input.BatchSize == output.BatchSize);

            for (var i = 0; i < actions.Length; i++)
            {
                GradientLogMuPolicy(input, mu, actions[i], startBatchIndex, output);
                startBatchIndex++;
            }
        }

        public void PolicyGradient(Vol input, Vol mu, int[][] pathActions, double[] rewards, Vol output)
        {
            Debug.Assert(input.BatchSize == pathActions.Sum(a => a.Length));
            Debug.Assert(pathActions.Length == rewards.Length);

            var likelihood = build.SameAs(input.Shape);

            int batchIndexOfPath = 0;
            for (var i = 0; i < pathActions.Length; i++)
            {
                likelihood.Clear();
                PathLikelihoodRatio(input, mu, pathActions[i], batchIndexOfPath, likelihood);
                likelihood.DoMultiply(likelihood, -rewards[i]);

                likelihood.DoAdd(output, output);

                batchIndexOfPath += pathActions[i].Length;
            }
        }

        //def discount_rewards(r):
        //""" take 1D float array of rewards and compute discounted reward """
        //discounted_r = np.zeros_like(r)
        //running_add = 0
        //for t in reversed(xrange(0, r.size)):
        //   running_add = running_add* gamma + r[t]
        //   discounted_r[t] = running_add
        //return discounted_r;

        private double[] DiscountedRewards(double[] rewards, double gamma)
        {
            double[] discounted = new double[rewards.Length];
            double running = Ops<double>.Zero;
            for (var i = rewards.Length - 1; i >= 0; i--)
            {
                running = Ops<double>.Multiply(running, Ops<double>.Cast(gamma));
                running = Ops<double>.Add(running, rewards[i]);
                discounted[i] = running;
            }
            return discounted;
        }

        public void SetLoss(int[][] pathActions, double[] rewards)
        {
            this.rewards = rewards;
            this.pathActions = pathActions;

            var totalBatchSize = this.pathActions.Sum(e => e.Length);

            if (totalBatchSize != this.InputActivationGradients.BatchSize)
                throw new NotSupportedException("Total number of actions must match batchSize!");

            if (rewards.Length != pathActions.Length)
                throw new NotSupportedException("Number of paths and rewards must match!");
        }

        protected override Volume<double> Forward(Volume<double> input, bool isTraining = false)
        {
            MuPolicy(input, OutputActivation);

            return OutputActivation;
        }

        public override void Backward(Volume<double> y)
        {
            Backward(y, out var unused);
        }

        public override void Backward(Volume<double> y, out double expectedReward)
        {
            var batches = this.OutputActivation.BatchSize;

            expectedReward = Ops<double>.Zero;
            int batch = 0;
            for (var path = 0; path < this.rewards.Length; path++)
            {
                var likelihood = 1.0;
                for (var action = 0; action < this.pathActions[path].Length; action++)
                {
                    likelihood *= OutputActivation.Get(0, 0, this.pathActions[path][action], batch);
                    batch++;
                }
                expectedReward = Ops<double>.Add(expectedReward, likelihood * rewards[path]);
            }

            this.InputActivationGradients.Clear();

            PolicyGradient(this.InputActivation, this.OutputActivation, this.pathActions, this.rewards, this.InputActivationGradients);
        }
    }
}