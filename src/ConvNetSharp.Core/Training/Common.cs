using ConvNetSharp.Volume;
using System.Linq;
using System.Diagnostics;
using ConvNetSharp.Core.Layers;
using System;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Training
{
    public class ActionInput
    {
        public int Action;
        public Volume<double> Inputs;

        public override string ToString()
        {
            return "action: " + Action.ToString();
        }
    }

    public class ActionInputReward
    {
        public int Action;
        public Volume<double> Inputs;
        public double Reward;

        public override string ToString()
        {
            return $"{Action} -> {Reward:0.000} reward";
        }
    }

    public class Path : List<ActionInputReward>
    {
        internal bool Used;

        public void SetReward(double reward)
        {
            foreach (var action in this)
                action.Reward = reward;
        }

        public void Add(ActionInput step)
        {
            this.Add(new ActionInputReward
            {
                Action = step.Action,
                Inputs = step.Inputs,
                Reward = 0
            });
        }

        public override string ToString()
        {
            var avg = this.Average(a => a.Reward);
            var actions = string.Join(", ", this.Select(e => e.Action).ToArray());
            return $"[{actions}] -> {avg:0.000} reward";
        }
    }
}
