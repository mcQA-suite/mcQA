from mcqa.data import MCQAData

def test_read_mcqa_examples(mcqa_data, dummy_data_path):
    examples = mcqa_data.read_mcqa_examples(dummy_data_path, 
                                            is_training=False)
    
    print(len(examples))
    assert len(examples) == 1

    examples = mcqa_data.read_mcqa_examples(dummy_data_path, 
                                            is_training=True)
    
    assert len(examples) == 1
    assert examples[0].label == "0"