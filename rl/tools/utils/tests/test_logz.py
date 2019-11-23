import logz

logz.configure_output_dir('test_logz_files')

# The first record.
logz.log_tabular('attr1', 1)
logz.log_tabular('attr2', 2)
logz.dump_tabular()

# The second record.
logz.log_tabular('attr1', 11)
logz.log_tabular('attr3', 13)
logz.dump_tabular()
