using System;

namespace ConvNetSharp
{
    public static class RandomUtilities
    {
        private static readonly Random Random = new Random(Seed);

        private static double val;
        private static bool returnVal;

        public static int Seed => (int) DateTime.Now.Ticks;

        public static double GaussianRandom()
        {
            if (returnVal)
            {
                returnVal = false;
                return val;
            }

            double r = 0, u = 0, v = 0;

            //System.Random is not threadsafe
            lock (Random)
            {
                while (r < double.Epsilon || r > 1)
                {
                    u = 2*Random.NextDouble() - 1;
                    v = 2*Random.NextDouble() - 1;
                    r = u*u + v*v;
                }
            }

            var c = Math.Sqrt(-2*Math.Log(r)/r);
            val = v*c; //cache this
            returnVal = true;

            return u*c;
        }

        public static double Randn(double mu, double std)
        {
            return mu + GaussianRandom()*std;
        }
    }
}