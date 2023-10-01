try:
	from datapipeline import DataPipeline
	from clusterpipeline import ClusterPipeline
	from portfoliopipeline import PortfolioPipeline
	from backtestpipeline import BacktestPipeline
except:
	pass

sectors_dict = {
	'final': ['DBCN UX9SXI5CAPNP', 'HEWG VNTW0AC8LAHX', 'GSJY W8L8B8ZCNXB9', 'ITF S96RH23DIAUD', 'EFO UD63CSAA26P1', 'UPV UM61FJMT8EHX', 'EFU TX34HT712KBP',  'EPV UDJVM3EN4QXX', 'DGZ U0K69ONGSDPH', 'DZZ U0J6TLAAPMJP'] #, 'GLL U85WJOCE24BP']
}

total_pairs_found = 0
total_pairs_validated = 0
final_results = pd.DataFrame({ 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [], 'Best entry z threshold': [], 'Best exit z threshold': [], 'Validation Fees': [], 'Validation PnL': [], 'Sector': [] })

for sector in sectors_dict:
	display(f'Doing {sector} sector now')

	training_df, validation_df, testing_df, training_and_validation_df = DataPipeline(sectors_dict[sector], datetime(2016, 1, 1), datetime(2023, 1, 1), datetime(2020, 1, 1), datetime(2022, 1, 1))

	clusters = [training_df.columns] if training_df.shape[1] >= 15 else ClusterPipeline(training_df) 

	portfolio_pipe = PortfolioPipeline()
	backtest_pipe = BacktestPipeline()

	# get pairs for validation test
	validation_pairs = portfolio_pipe.find_pairs_from_clusters(training_df, clusters)
	if validation_pairs is None or validation_pairs.shape[0] == 0:
		display(f'Found nothing in {sector} sector')
		continue

	total_pairs_found += validation_pairs.shape[0]

	initial_capital = 30000
	# initial_capital_per_pair = initial_capital // validation_pairs.shape[0]

	# validation backtest
	validation_backtest_results = backtest_pipe.validation_backtest(training_df, validation_df, validation_pairs, initial_capital=5000)
	sorted_validation_backtest_results = sorted(validation_backtest_results, key=lambda x: x[-1][-1]-x[-1][0], reverse=True)
	backtest_pipe.plot_validation_backtest_results(sorted_validation_backtest_results)

	filtered_validation_pairs = portfolio_pipe.filter_positive_pnl_pairs(validated_pairs, sorted_validation_backtest_results, min_validation_return_threshold=0.1)
	if filtered_validated_pairs.shape[0] == 0:
		display(f'No positive validated pairs in {sector} sector')
		continue

	total_pairs_validated += filtered_validated_pairs.shape[0]

	# get final test pairs
	test_pairs = portfolio_pipe.revalidate_pairs(training_and_validation_df, filtered_validated_pairs, sectors)

	# test backtest
	test_backtest_results = backtest_pipe.test_backtest(training_and_validation_df, testing_df, test_pairs, initial_capital=5000)


	final_results = pd.concat([final_results, test_pairs], ignore_index=True)

display(final_results)
display(f'Final result for all sectors: Total pairs validated / total pairs found = {total_pairs_validated}/{total_pairs_found} = {total_pairs_validated/total_pairs_found*100:.2f}% pairs ')