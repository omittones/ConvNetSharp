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
        private int[][] pathActions;
        private double gamma;
        private double[] returns;
        private Vol maxes;

        public ReinforcementLayer()
        {
        }

        public override LayerBase<double> Clone()
        {
            return new ReinforcementLayer(); 
        }

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

        public void Policy(Vol input, Vol output)
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

        public void LogPolicy(Vol input, Vol output)
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

        public void GradientLogPolicy(Vol input, Vol pi, int action, double scaling, int batch, Vol output)
        {
            Debug.Assert(input.Depth == output.Depth);
            Debug.Assert(input.BatchSize == output.BatchSize);

            for (var d = 0; d < input.Depth; d++)
            {
                var flag = d == action ? 1.0 : 0.0;
                var gradient = flag - pi.Get(0, 0, d, batch);
                output.Set(0, 0, d, batch, gradient * scaling);
            }
        }

        //public void GradientPolicy(Vol input, Vol pi, int[][] pathActions, double[] pathReturns, Vol output)
        //{
        //    Debug.Assert(input.BatchSize == pathActions.Sum(a => a.Length));
        //    Debug.Assert(pathActions.Length == pathReturns.Length);

        //    int batchIndexOfPath = 0;
        //    for (var i = 0; i < pathActions.Length; i++)
        //    {
        //        var tdRewards = TimeDiscountedRewards(pathActions[i].Length, pathReturns[i]);

        //        PathLikelihoodRatio(input, pi, pathActions[i], tdRewards, batchIndexOfPath, output);

        //        batchIndexOfPath += pathActions[i].Length;
        //    }
        //}

        //public void PathLikelihoodRatio(Vol input, Vol pi, int[] actions, double[] scaling, int startBatchIndex, Vol output)
        //{
        //    Debug.Assert(input.Depth == output.Depth);
        //    Debug.Assert(input.BatchSize == output.BatchSize);

        //    for (var i = 0; i < actions.Length; i++)
        //    {
        //        GradientLogPolicy(input, pi, actions[i], scaling[i], startBatchIndex, output);
        //        startBatchIndex++;
        //    }
        //}

        //private double[] TimeDiscountedRewards(int length, double pathReturn)
        //{
        //    double[] discounted = new double[length];
        //    double running = Ops<double>.Zero;
        //    for (var i = length - 1; i >= 0; i--)
        //    {
        //        running = Ops<double>.Multiply(running, gamma);
        //        running = Ops<double>.Add(running, pathReturn);
        //        discounted[i] = running;
        //    }
        //    return discounted;
        //}

        //public void SetReturns(int[][] pathActions, double[] returns)
        //{
        //    this.returns = returns;

        //    this.pathActions = pathActions;

        //    var totalBatchSize = this.pathActions.Sum(e => e.Length);

        //    if (totalBatchSize != this.InputActivationGradients.BatchSize)
        //        throw new NotSupportedException("Total number of actions must match batchSize!");

        //    if (returns.Length != pathActions.Length)
        //        throw new NotSupportedException("Number of paths and rewards must match!");
        //}

        protected override Volume<double> Forward(Volume<double> input, bool isTraining = false)
        {
            if (this.maxes == null || this.maxes.BatchSize != input.BatchSize)
                this.maxes = build.SameAs(1, 1, 1, input.BatchSize);

            Policy(input, OutputActivation);

            return OutputActivation;
        }

        public override void Backward(Volume<double> y)
        {
            Backward(y, out var unused);
        }

        public override void Backward(Volume<double> y, out double estimatedLoss)
        {
            estimatedLoss = Ops<double>.Zero;
            //int batch = 0;
            //for (var path = 0; path < this.returns.Length; path++)
            //{
            //    var nmActions = this.pathActions[path].Length;
            //    for (var action = 0; action < nmActions; action++)
            //    {
            //        var expectation = OutputActivation.Get(0, 0, this.pathActions[path][action], batch) * returns[path];
            //        estimatedLoss -= expectation;
            //        batch++;
            //    }
            //}

            Debug.Assert(y.BatchSize == 1);

            this.InputActivationGradients.Clear();

            //TODO - optimize this!
            var klass = y.IndexOfMax();
            this.GradientLogPolicy(this.InputActivation, this.OutputActivation, klass, 1.0, 0, this.InputActivationGradients);

            //TODO - gradient ascent instead of descent, so do this on trainer level
            this.InputActivationGradients.DoNegate(this.InputActivationGradients);

            Ops<double>.Validate(this.InputActivationGradients);
        }
    }
}