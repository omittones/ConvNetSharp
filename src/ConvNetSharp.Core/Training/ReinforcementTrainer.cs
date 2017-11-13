using ConvNetSharp.Volume;
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Training
{
    public class ActionGradient
    {
        public int Action;
        public Volume<double> Inputs;
    }
    
    public class ActionGradientReward
    {
        public int Action;
        public Volume<double> Inputs;
        public double Reward;
    }

    public class Path : List<ActionGradientReward>
    {
        internal bool Used;

        public void SetReward(double reward, double gamma = 1.0)
        {
            double previous = 0;
            foreach (var ag in this)
            {
                ag.Reward = previous * gamma + reward;
                previous = ag.Reward;
            }
        }

        public void Add(ActionGradient step)
        {
            this.Add(new ActionGradientReward
            {
                Action = step.Action,
                Inputs = step.Inputs,
                Reward = 0
            });
        }
    }
    
    public class ReinforcementTrainer : SgdTrainer<double>
    {
        public bool ApplyBaselineAndNormalizeReturns { get; set; }

        public double Baseline { get; private set; }
        public double RewardDiscountGamma { get; set; }
        public double EstimatedRewards => Ops<double>.Negate(this.Loss);

        private readonly SoftmaxLayer<double> finalLayer;
        private readonly InputLayer<double> inputLayer;
        private Volume<double> input;
        private Volume<double> output;

        public ReinforcementTrainer(Net<double> net) : base(net)
        {
            this.Baseline = Ops<double>.Zero;
            this.RewardDiscountGamma = Ops<double>.Zero;
            this.ApplyBaselineAndNormalizeReturns = true;

            this.inputLayer = net.Layers
                .OfType<InputLayer<double>>()
                .Single();

            this.finalLayer = net.Layers
                .OfType<SoftmaxLayer<double>>()
                .Last();
        }

        public ActionGradient Act(Volume<double> inputs)
        {
            if (inputs.BatchSize != 1)
                throw new NotSupportedException("Not supported!");

            var output = this.Net.Forward(inputs, false);
            Debug.Assert(output.Width == 1);
            Debug.Assert(output.Height == 1);
            Debug.Assert(output.BatchSize == 1);

            var action = output.IndexOfMax();

            return new ActionGradient
            {
                Inputs = inputs.Clone(),
                Action = action
            };
        }

        private void ReinitCache(int batchSize)
        {
            if (this.input == null ||
                this.input.BatchSize != batchSize)
            {
                this.output = BuilderInstance<double>.Volume.SameAs(
                    finalLayer.InputWidth,
                    finalLayer.InputHeight,
                    finalLayer.InputDepth,
                    batchSize);
                this.input = BuilderInstance<double>.Volume.SameAs(
                    inputLayer.InputWidth,
                    inputLayer.InputHeight,
                    inputLayer.InputDepth,
                    batchSize);
            }
        }

        public void Reinforce(Path[] paths)
        {
            var netGrads = this.Net.GetParametersAndGradients();
            foreach (var png in netGrads)
                png.Gradient.Clear();

            this.BatchSize = paths.Select(p => p.Count).Sum();
            ReinitCache(this.BatchSize);

            double average = 0.0;

            output.Clear();
            var flatInput = input.ReShape(1, 1, -1, this.BatchSize);
            this.finalLayer.BatchRewards = new double[this.BatchSize];
            int currentBatch = 0;
            foreach (var path in paths)
            {
                if (path.Used)
                    throw new NotSupportedException();

                foreach (var action in path)
                {
                    var flatAction = action.Inputs.ReShape(1, 1, -1, 1);
                    for (var d = 0; d < flatAction.Depth; d++)
                        flatInput.Set(0, 0, d, currentBatch, flatAction.Get(0, 0, d, 0));

                    output.Set(0, 0, action.Action, currentBatch, 1.0);

                    this.finalLayer.BatchRewards[currentBatch] = action.Reward;

                    average += (action.Reward / this.BatchSize);

                    currentBatch++;
                }

                path.Used = true;
            }

            this.Loss = average;

            //apply baseline
            for (var i = 0; i < this.BatchSize; i++)
                this.finalLayer.BatchRewards[i] = this.finalLayer.BatchRewards[i] - average;

            this.Forward(input);
            
            this.Backward(output);

            //gradient ascent!
            foreach (var grad in this.Net.GetParametersAndGradients())
                grad.Gradient.DoNegate(grad.Gradient);

            //if (ApplyBaselineAndNormalizeReturns)
            //{
            //    returns = returns
            //        .Select(r => r - Baseline)
            //        .ToArray();
            //    //normalize by standard deviation
            //    var stdDev = returns
            //        .Select(a => a * a)
            //        .Average();
            //    returns = returns
            //        .Select(r => r / stdDev)
            //        .ToArray();
            //}
            //if (this.advantage == null || this.advantage.Length != returns.Length)
            //    this.advantage = new double[returns.Length];
            //this.baseline = baseline;
            ////advantage = discounted returns - baseline
            ////normalize by stddev
            //double stddev = 0;
            //for (var i = 0; i < returns.Length; i++)
            //{
            //    var adv = Ops<double>.Subtract(returns[i], baseline);
            //    stddev += (adv * adv) / returns.Length;
            //    this.advantage[i] = adv;
            //}
            //stddev = Ops<double>.Sqrt(stddev);
            //if (stddev != 0)
            //    for (var i = 0; i < advantage.Length; i++)
            //        this.advantage[i] = this.advantage[i] / stddev;
            //this.finalLayer.SetReturns(pathActions, returns);
            //if (ApplyBaselineAndNormalizeReturns)
            //{
            //    //refit baseline to minimize Sum[(R - b)^2]
            //    Baseline = Ops<double>.Zero;
            //    for (var i = 0; i < pathReturns.Length; i++)
            //        Baseline = Ops<double>.Add(pathReturns[i], Baseline);
            //    Baseline = Ops<double>.Divide(Baseline, Ops<double>.Cast(pathReturns.Length));
            //}
            //this.Backward(pathInputs);

            var chrono = Stopwatch.StartNew();

            TrainImplem();

            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / paths.Sum(p => p.Count);
        }

        public override void Train(Volume<double> x, Volume<double> y)
        {
            throw new NotSupportedException("Use Reinforce method instead!");
        }
    }
}
