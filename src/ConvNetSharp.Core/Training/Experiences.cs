using System;
using System.Collections;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Training
{
    internal class Experiences : IReadOnlyList<Experience>
    {
        public int Size { get; set; }
        public ExperienceDiscardStrategy DiscardStrategy { get; set; }
        public int Count => inner.Count;
        public bool IsReadOnly => false;

        private List<Experience> inner;
        private int indexOfMax = 0;
        private int indexOfMin = 0;

        public Experiences()
        {
            this.inner = new List<Experience>();
        }

        public Experience this[int index]
        {
            get => inner[index];
        }

        public void Add(Experience experience)
        {
            if (this.inner.Count == Size)
            {
                if (DiscardStrategy == ExperienceDiscardStrategy.First)
                    this.inner.RemoveAt(0);
                else if (DiscardStrategy == ExperienceDiscardStrategy.BestReward)
                    this.inner.RemoveAt(indexOfMax);
                else if (DiscardStrategy == ExperienceDiscardStrategy.WorstReward)
                    this.inner.RemoveAt(indexOfMin);

                this.inner.Add(experience);

                RebuildMinMax();
            }
            else
            {
                this.inner.Add(experience);

                if (this.inner[indexOfMin].reward > experience.reward)
                    indexOfMin = this.inner.Count - 1;
                if (this.inner[indexOfMax].reward < experience.reward)
                    indexOfMax = this.inner.Count - 1;
            }
        }

        private void RebuildMinMax()
        {
            indexOfMin = 0;
            indexOfMax = 0;
            for (var i = 1; i < this.inner.Count; i++)
                if (this.inner[i].reward > this.inner[indexOfMax].reward)
                    indexOfMax = i;
            for (var i = 1; i < this.inner.Count; i++)
                if (this.inner[i].reward < this.inner[indexOfMin].reward)
                    indexOfMin = i;
        }

        public int IndexOf(Experience item)
        {
            return inner.IndexOf(item);
        }

        public IEnumerator<Experience> GetEnumerator()
        {
            return inner.GetEnumerator();
        }

        internal void Clear()
        {
            inner.Clear();
            indexOfMax = 0;
            indexOfMin = 0;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return inner.GetEnumerator();
        }
    }
}
