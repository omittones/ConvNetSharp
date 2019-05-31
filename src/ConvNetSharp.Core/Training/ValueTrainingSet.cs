using ConvNetSharp.Volume;
using System;
using System.Linq;

namespace ConvNetSharp.Core.Training
{
    public class ValueTrainingSet
    {
        private int Index;
        public Volume<double> Inputs;
        public Volume<double> Outputs;

        private ValueTrainingSet(Path[] paths)
        {
            var stateShape = paths.First().First().State.Shape;
            var batchSize = paths.Sum(p => p.Count * stateShape.Dimensions[3]);
            stateShape.SetDimension(3, batchSize);

            this.Index = 0;
            this.Inputs = BuilderInstance<double>.Volume.SameAs(stateShape);
            this.Outputs = BuilderInstance<double>.Volume.SameAs(1, 1, 1, batchSize);
        }

        public static ValueTrainingSet CreateWithEstimator(Path[] paths, double gamma, Net<double> valueEstimator)
        {
            var set = new ValueTrainingSet(paths);
            ActionInputReward lastAction = null;
            foreach (var path in paths)
                foreach (var action in path)
                {
                    if (lastAction != null)
                    {
                        var output = valueEstimator.Forward(action.State);
                        set.SetStateEstimate(lastAction.State, action.Reward + gamma * output.Get(0, 0, 0, 0));
                    }
                    lastAction = action;
                }
            if (set.Index != set.Inputs.BatchSize)
                throw new NotSupportedException();
            set.Index = 0;
            return set;
        }

        public static ValueTrainingSet CreateWithRewardToGo(Path[] paths)
        {
            var set = new ValueTrainingSet(paths);
            foreach (var path in paths)
                set.LoadFromPathUsingRewardsToGo(path);
            if (set.Index != set.Inputs.BatchSize)
                throw new NotSupportedException();
            set.Index = 0;
            return set;
        }
        
        private void LoadFromPathUsingRewardsToGo(Path path)
        {
            var rewardsToGo = new double[path.Count];
            rewardsToGo[path.Count - 1] = path.Last().Reward;
            for (var i = path.Count - 2; i >= 0; i++)
                rewardsToGo[i] = path[i].Reward + rewardsToGo[i + 1];

            for (var i = 0; i < path.Count; i++)
                SetStateEstimate(path[i].State, rewardsToGo[i]);
        }

        private void SetStateEstimate(Volume<double> state, double valueEstimate)
        {
            state.CopyTo(Inputs, this.Index);
            this.Index += Inputs.BatchSize;
            this.Outputs.Set(0, 0, 0, this.Index, valueEstimate);
        }
    }
}
