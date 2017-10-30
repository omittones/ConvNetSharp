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
        private int indexOfBest = 0;
        private int indexOfWorst = 0;
        private int indexOfNext = 0;

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
                {
                    this.inner[indexOfNext] = experience;
                    indexOfNext++;
                    if (indexOfNext == this.inner.Count)
                        indexOfNext = 0;
                }
                else
                {
                    if (DiscardStrategy == ExperienceDiscardStrategy.BestReward)
                        this.inner[indexOfBest] = experience;
                    else if (DiscardStrategy == ExperienceDiscardStrategy.WorstReward)
                        this.inner[indexOfWorst] = experience;
                    RebuildIndexes();
                }
            }
            else
            {
                this.inner.Add(experience);

                if (this.inner[indexOfWorst].reward > experience.reward)
                    indexOfWorst = this.inner.Count - 1;
                if (this.inner[indexOfBest].reward < experience.reward)
                    indexOfBest = this.inner.Count - 1;
            }
        }

        private void RebuildIndexes()
        {
            indexOfWorst = 0;
            indexOfBest = 0;
            for (var i = 1; i < this.inner.Count; i++)
            {
                if (this.inner[i].reward > this.inner[indexOfBest].reward)
                    indexOfBest = i;
                if (this.inner[i].reward < this.inner[indexOfWorst].reward)
                    indexOfWorst = i;
            }
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
            indexOfBest = 0;
            indexOfWorst = 0;
            indexOfNext = 0;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return inner.GetEnumerator();
        }
    }
}
