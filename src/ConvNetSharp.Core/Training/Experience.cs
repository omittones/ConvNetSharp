using ConvNetSharp.Core.Serialization;
using System.Linq;

namespace ConvNetSharp.Core.Training
{
    internal class Experience
    {
        public double reward;
        public double[] state;
        public double[] nextState;
        public int actionTaken;

        internal static Experience New(double[] s0, int a0, double r0, double[] s1)
        {
            return new Experience
            {
                state = s0,
                nextState = s1,
                actionTaken = a0,
                reward = r0
            };
        }

        public override string ToString()
        {
            return $"({state.ToHumanString("{0:0.000}")},{actionTaken}) -> ({nextState.ToHumanString("{0:0.000}")}) with reward({reward:0.0000})";
        }
    }
}
