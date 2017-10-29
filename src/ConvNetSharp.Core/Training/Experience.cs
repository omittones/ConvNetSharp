namespace ConvNetSharp.Core.Training
{
    internal struct Experience
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
    }
}
