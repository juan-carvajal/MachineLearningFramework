using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ErrorFunction
{
    class CrossEntropy : ErrorFunction
    {
        public override double GetDerivativeValue(double label, double estimator)
        {
            if (label == 1)
            {
                return -1.0 / estimator;
            }
            else
            {
                return 1.0/(1-estimator);
            }
        }

        public override double GetValue(double label, double estimator)
        {
            if (label == 1)
            {
                return -Math.Log(estimator);
            }
            else
            {
                return -Math.Log(1 - estimator);
            }
        }
    }
}
