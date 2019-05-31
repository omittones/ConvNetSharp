﻿using ConvNetSharp.Volume;

namespace ConvNetSharp.ReinforcementLearning.Deep
{
    /// <summary>
    /// An experience stores the initial state of an agent, the taken action, the received reward and the final state of the agent.
    /// </summary>
    public class Experience
    {
        #region Member Fields
        private Volume<double> _initialState;
        private int _action;
        private double _reward;
        private Volume<double> _finalState;
        #endregion

        #region Member Properties
        public Volume<double> InitialState
        {
            get { return _initialState; }
        }

        public int Action
        {
            get { return _action; }
        }

        public double Reward
        {
            get { return _reward; }
        }

        public Volume<double> FinalState
        {
            get { return _finalState; }
        }
        #endregion

        #region Constructor
        public Experience(Volume<double> initialState, int initialAction, double initialReward, Volume<double> finalState)
        {
            _initialState = initialState;
            _action = initialAction;
            _reward = initialReward;
            _finalState = finalState;
        }
        #endregion
    }
}