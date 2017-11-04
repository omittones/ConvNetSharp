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
        private double[] advantage;
        private int[][] pathActions;
        private double gamma;
        private double[] returns;
        private double baseline;
        private Vol maxes;

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
            input.DoMax(this.maxes);
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
            input.DoMax(this.maxes);
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

        public void GradientLogMuPolicy(Vol input, Vol mu, int action, double scaling, int batch, Vol output)
        {
            Debug.Assert(input.Depth == output.Depth);
            Debug.Assert(input.BatchSize == output.BatchSize);

            for (var d = 0; d < input.Depth; d++)
            {
                var flag = d == action ? 1.0 : 0.0;
                var gradient = flag - mu.Get(0, 0, d, batch);
                output.Set(0, 0, d, batch, gradient * scaling);
            }
        }

        public void PathLikelihoodRatio(Vol input, Vol mu, int[] actions, double[] scaling, int startBatchIndex, Vol output)
        {
            Debug.Assert(input.Depth == output.Depth);
            Debug.Assert(input.BatchSize == output.BatchSize);

            for (var i = 0; i < actions.Length; i++)
            {
                GradientLogMuPolicy(input, mu, actions[i], scaling[i], startBatchIndex, output);
                startBatchIndex++;
            }
        }

        public void PolicyGradient(Vol input, Vol mu, int[][] pathActions, double[] pathReturns, Vol output)
        {
            Debug.Assert(input.BatchSize == pathActions.Sum(a => a.Length));
            Debug.Assert(pathActions.Length == pathReturns.Length);

            int batchIndexOfPath = 0;
            for (var i = 0; i < pathActions.Length; i++)
            {
                var tdRewards = TimeDiscountedRewards(pathActions[i].Length, pathReturns[i]);

                PathLikelihoodRatio(input, mu, pathActions[i], tdRewards, batchIndexOfPath, output);

                batchIndexOfPath += pathActions[i].Length;
            }
        }

        private double[] TimeDiscountedRewards(int length, double pathReturn)
        {
            double[] discounted = new double[length];
            double running = Ops<double>.Zero;
            for (var i = length - 1; i >= 0; i--)
            {
                running = Ops<double>.Multiply(running, gamma);
                running = Ops<double>.Add(running, pathReturn);
                discounted[i] = running;
            }
            return discounted;
        }

        public void SetReturns(int[][] pathActions, double[] returns, double baseline, double gamma)
        {
            if (this.advantage == null || this.advantage.Length != returns.Length)
                this.advantage = new double[returns.Length];
            this.returns = returns;
            this.baseline = baseline;

            //advantage = discounted returns - baseline
            //normalize by stddev
            double stddev = 0;
            for (var i = 0; i < returns.Length; i++)
            {
                var adv = Ops<double>.Subtract(returns[i], baseline);
                stddev += (adv * adv) / returns.Length;
                this.advantage[i] = adv;
            }
            stddev = Ops<double>.Sqrt(stddev);
            if (stddev != 0)
                for (var i = 0; i < advantage.Length; i++)
                    this.advantage[i] = this.advantage[i] / stddev;

            this.pathActions = pathActions;

            this.gamma = gamma;

            var totalBatchSize = this.pathActions.Sum(e => e.Length);

            if (totalBatchSize != this.InputActivationGradients.BatchSize)
                throw new NotSupportedException("Total number of actions must match batchSize!");

            if (returns.Length != pathActions.Length)
                throw new NotSupportedException("Number of paths and rewards must match!");
        }

        protected override Volume<double> Forward(Volume<double> input, bool isTraining = false)
        {
            if (this.maxes == null || this.maxes.BatchSize != input.BatchSize)
                this.maxes = build.SameAs(1, 1, 1, input.BatchSize);

            MuPolicy(input, OutputActivation);

            return OutputActivation;
        }

        public override void Backward(Volume<double> y)
        {
            Backward(y, out var unused);
        }

        public override void Backward(Volume<double> y, out double estimatedLoss)
        {
            estimatedLoss = Ops<double>.Zero;
            int batch = 0;
            for (var path = 0; path < this.advantage.Length; path++)
            {
                var nmActions = this.pathActions[path].Length;
                for (var action = 0; action < nmActions; action++)
                {
                    var expectation = OutputActivation.Get(0, 0, this.pathActions[path][action], batch) * returns[path];
                    estimatedLoss -= expectation;
                    batch++;
                }
            }

            this.InputActivationGradients.Clear();

            PolicyGradient(this.InputActivation, this.OutputActivation, this.pathActions, this.advantage, this.InputActivationGradients);

            this.InputActivationGradients.DoNegate(this.InputActivationGradients);

            Ops<double>.Validate(this.InputActivationGradients);
        }
    }
}