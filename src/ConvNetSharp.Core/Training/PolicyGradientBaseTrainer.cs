using ConvNetSharp.Volume;
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;

namespace ConvNetSharp.Core.Training
{
    public abstract class PolicyGradientBaseTrainer : SgdTrainer<double>
    {
        public double EstimatedRewards { get; protected set; }

        private readonly SoftmaxLayer<double> finalLayer;
        private readonly InputLayer<double> inputLayer;
        private Volume<double> input;
        private Volume<double> output;

        public PolicyGradientBaseTrainer(Net<double> net) : base(net)
        {
            this.inputLayer = net.Layers
                .OfType<InputLayer<double>>()
                .Single();

            this.finalLayer = net.Layers
                .OfType<SoftmaxLayer<double>>()
                .Last();
        }

        public virtual ActionInput Act(Volume<double> inputs)
        {
            if (inputs.BatchSize != 1)
                throw new NotSupportedException("Not supported!");

            var output = this.Net.Forward(inputs, false);
            Debug.Assert(output.Width == 1);
            Debug.Assert(output.Height == 1);
            Debug.Assert(output.BatchSize == 1);

            var action = output.IndexOfMax();

            return new ActionInput
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

        protected abstract double[] GetGradientMultipliers(Path[] paths);

        public virtual void Reinforce(Path[] paths)
        {
            var netGrads = this.Net.GetParametersAndGradients();
            foreach (var png in netGrads)
                png.Gradient.Clear();

            var batchSize = paths.Select(p => p.Count).Sum();
            ReinitCache(batchSize);

            this.BuildSets(paths);

            this.finalLayer.GradientMultiplier = this.GetGradientMultipliers(paths);

            this.Forward(input);

            this.Backward(output);

            //gradient ascent!
            foreach (var grad in this.Net.GetParametersAndGradients())
                grad.Gradient.Negate(grad.Gradient);

            var chrono = Stopwatch.StartNew();
            TrainImplem(batchSize);
            this.UpdateWeightsTimeMs = chrono.Elapsed.TotalMilliseconds / paths.Sum(p => p.Count);
        }

        private void BuildSets(Path[] paths)
        {
            output.Clear();

            var flatInput = input.ReShape(1, 1, -1, input.BatchSize);
            int currentBatch = 0;
            foreach (var path in paths)
            {
                if (path.Used)
                    throw new NotSupportedException();
                foreach (var action in path)
                {
                    var flatAction = action.State.ReShape(1, 1, -1, 1);
                    for (var d = 0; d < flatAction.Depth; d++)
                        flatInput.Set(0, 0, d, currentBatch, flatAction.Get(0, 0, d, 0));
                    output.Set(0, 0, action.Action, currentBatch, 1.0);
                    currentBatch++;
                }
                path.Used = true;
            }
        }

        public override Volume<double> Train(Volume<double> x, Volume<double> y)
        {
            throw new NotSupportedException("Use Reinforce method instead!");
        }
    }
}
