import unittest
from math import nan
from unittest.mock import patch
from unittest.mock import Mock
from unittest import TestCase
import time

from app import *

N = 10


class AppTests(TestCase):
    def setUp(self):
        self.app = App(testing=True)

    @patch('builtins.input', return_value='8')
    def test_get_number(self,s):
        self.assertEqual(self.app.get_number(""),8)

    @patch('app.App.get_number', return_value=2017)
    def test_get_year(self, get_number):
        self.assertEqual(self.app.get_year(),2017)

    @patch('app.App.get_number', return_value=2)
    def test_get_month(self, get_number):
        self.assertEqual(self.app.get_month(),2)

    @patch('app.App.get_number', return_value=3)
    def test_get_day(self, get_number):
        self.assertEqual(self.app.get_day(),3)

    @patch('app.App.get_number', return_value=20)
    def test_get_epochs(self, get_number):
        self.assertEqual(self.app.get_epochs(),20)

    @patch('app.App.get_number', return_value=3)
    def test_get_crypto(self, get_number):
        self.assertEqual(self.app.get_crypto(),'LTC-USD')

    @patch('app.App.get_number', return_value=2)
    def test_get_network(self, get_number):
        self.assertEqual(self.app.get_network(),'LSTM')


class QuantTests(TestCase):
    def setUp(self):
        self.q = Quant()

    def test_add_transition(self):
        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.assertEqual(len(self.q.data),1)

    def test_sample_batches(self):
        for _ in range(N):
            self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.assertEqual(len(self.q.data),N)

    def test_compute_feature(self):
        test_data = [
                [673.88, 674.08, 600.60, 608.69, 58.1],
                [678.09, 681.94, 671.17, 672.74, 31.3],
                [677.76, 682.56, 661.88, 677.55, 36.4],
                [674.39, 682.50, 670.92, 676.94, 40.3],
                [651.07, 678.14, 644.68, 673.82, 42.0],
                [654.53, 658.11, 649.27, 649.27, 30.2],
                [645.45, 659.49, 640.29, 655.16, 31.7],
                [667.08, 668.08, 643.87, 645.58, 33.9],
                [634.75, 670.14, 633.46, 667.35, 57.1],
                ]
        test_feature = [21.769999999999982, 32.60000000000002, 0.04185699607345483, 1.6851004330336108, -1, -1, -1, -1]

        self.q.data = test_data
        self.assertCountEqual(self.q.compute_feature(), test_feature)

    def test_get_signal_history(self):
        for i in range(N):
            self.q.add_transition([1.,1.,1.,1.,1.],i,1,1)
        self.assertCountEqual(self.q.get_signal_history(), [i for i in range(N)])

    def test_get_data_history(self):
        for i in range(N):
            self.q.add_transition([i-2.,i-1.,i+1.,i+2.,i+3.],1,1,1)
        self.assertCountEqual(self.q.get_data_history(), [[i-2.,i-1.,i+1.,i+2.,i+3.] for i in range(N)])

    def test_get_last_action(self):
        for i in range(N):
            self.q.add_transition([1.,1.,1.,1.,1.],i,1,1)
        self.assertEqual(self.q.get_last_action()[0], N-1)

    def test_get_total_return(self):
        for i in range(N):
            self.q.add_transition([1.,1.,1.,1.,1.],1,i,1)
        self.assertEqual(self.q.get_total_return(), N*(N-1)/2)

    def test_get_return(self):
        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.assertEqual(self.q.get_return(0,[1.,1.,1.,1.,1.]), -0.01)
        self.assertEqual(self.q.get_return(0,[0.,1.,1.,0.,1.]), 0.0)
        self.assertEqual(self.q.get_return(0,[2.,1.,1.,2.,1.]), -0.02)
        self.assertEqual(self.q.get_return(1,[1.,1.,1.,1.,1.]), 0.0)
        self.assertEqual(self.q.get_return(1,[0.,1.,1.,0.,1.]), -100.0)
        self.assertEqual(self.q.get_return(1,[2.,1.,1.,2.,1.]), 100.0)
        self.assertEqual(self.q.get_return(2,[1.,1.,1.,1.,1.]), -0.01)
        self.assertEqual(self.q.get_return(2,[0.,1.,1.,0.,1.]), 100.0)
        self.assertEqual(self.q.get_return(2,[2.,1.,1.,2.,1.]), -100.02)

    def test_get_last_trend(self):
        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.assertEqual(self.q.get_last_trend(), 0)

        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.q.add_transition([2.,1.,1.,2.,1.],1,1,1)
        self.assertEqual(self.q.get_last_trend(), 1)

        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.q.add_transition([0.,1.,1.,0.,1.],1,1,1)
        self.assertEqual(self.q.get_last_trend(), 2)

    def test_eval_performance(self):
        test_data = [
                [673.88, 674.08, 600.60, 608.69, 58.1],
                [678.09, 681.94, 671.17, 672.74, 31.3],
                [677.76, 682.56, 661.88, 677.55, 36.4],
                [674.39, 682.50, 670.92, 676.94, 40.3],
                [651.07, 678.14, 644.68, 673.82, 42.0],
                [654.53, 658.11, 649.27, 649.27, 30.2],
                [645.45, 659.49, 640.29, 655.16, 31.7],
                [667.08, 668.08, 643.87, 645.58, 33.9],
                [634.75, 670.14, 633.46, 667.35, 57.1],
                ]
        test_eval = [5836.951300000019, 0.24826534687950222, 7.7388, 0.2430084704169348]

        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        self.q.add_transition([1.,1.,1.,1.,1.],1,1,1)
        for s in test_data:
            a = self.q.get_last_trend()
            self.q.add_transition(s, a, self.q.get_return(a,s), 0)
        self.assertCountEqual(self.q.eval_performance(), test_eval)


class UtilsTests(TestCase):
    def test_convert_nan(self):
        self.assertEqual(convert_nan([1,2,nan,3,nan,44,5,nan]), [1,2,-1,3,-1,44,5,-1])

    def test_vectorize(self):
        self.assertCountEqual(vectorize(5,3),[0,0,0,1,0])


class BrokerTests(TestCase):
    def setUp(self):
        self.b = Broker()
        self.granularity = 3600

    def test_make_order_0(self):
        self.b.make_order(day=1, month=1, year=2000)
        self.assertEqual(self.b.data.empty, True)
        self.b.make_order(day=27, month=12, year=2017)
        self.assertEqual(self.b.data.empty, False)

    def test_make_order_1(self):
        self.b.product_id = 'BTC-USD'
        self.b.make_order(day=1, month=1, year=2016)
        self.assertEqual(self.b.data.empty, False)
        self.b.make_order(day=27, month=4, year=2018)
        self.assertEqual(self.b.data.empty, False)

    def test_make_order_2(self):
        self.b.product_id = 'ETH-USD'
        self.b.make_order(day=1, month=1, year=2017)
        self.assertEqual(self.b.data.empty, False)
        self.b.make_order(day=27, month=4, year=2018)
        self.assertEqual(self.b.data.empty, False)

    def test_make_order_3(self):
        self.b.product_id = 'LTC-USD'
        self.b.make_order(day=1, month=1, year=2017)
        self.assertEqual(self.b.data.empty, False)
        self.b.make_order(day=27, month=4, year=2018)
        self.assertEqual(self.b.data.empty, False)


if __name__ == '__main__':
    unittest.main()
