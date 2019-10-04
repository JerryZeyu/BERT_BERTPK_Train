export WordTree_DIR=/home/zeyuzhang/PycharmProjects/BERT_BERTPK_Train/BERT_BERTPK_train_5/expl-tablestore-export-2019-09-10-165215
export Step2_Model_DIR=/home/zeyuzhang/PycharmProjects/BERT_BERTPK_Train/BERT_BERTPK_train_5/step2/step2_model
export Step1_Output_DIR=/home/zeyuzhang/PycharmProjects/BERT_BERTPK_Train/BERT_BERTPK_train_5/step1/step1_output
export Step2_Output_DIR=/home/zeyuzhang/PycharmProjects/BERT_BERTPK_Train/BERT_BERTPK_train_5/step2/step2_output
export Train_DIR=/home/zeyuzhang/PycharmProjects/BERT_BERTPK_Train/BERT_BERTPK_train_5/expl-tablestore-export-2019-09-10-165215/question_train_5.tsv
export Dev_DIR=/home/zeyuzhang/PycharmProjects/BERT_BERTPK_Train/BERT_BERTPK_train_5/expl-tablestore-export-2019-09-10-165215/question_dev_5.tsv
export TASK_NAME=EPRG

python step1/run_produce_shortlist.py \
          --model_type bert \
            --model_name_or_path bert-base-cased \
              --task_name $TASK_NAME \
	       --do_train \
                  --do_eval \
                    --do_lower_case \
                      --data_dir $WordTree_DIR \
                       --train_data_dir $Train_DIR \
                        --dev_data_dir $Dev_DIR \
                        --max_seq_length 128 \
                          --per_gpu_train_batch_size 32 \
                          --per_gpu_eval_batch_size 32 \
                            --learning_rate 2e-5 \
                              --num_train_epochs 3.0 \
                               --step1_output_dir $Step1_Output_DIR \
                                --output_dir $Step1_Model_DIR/$TASK_NAME/

python step2/run_PK_output.py \
	  --model_type bert \
	    --model_name_or_path bert-base-cased \
	      --task_name $TASK_NAME \
	        --do_train \
		  --do_eval \
		    --do_lower_case \
		      --data_dir $WordTree_DIR \
		      --train_data_dir $Train_DIR \
		        --dev_data_dir $Dev_DIR \
		        --max_seq_length 128 \
			  --per_gpu_train_batch_size 32 \
			  --per_gpu_eval_batch_size 32 \
			    --learning_rate 2e-5 \
			      --num_train_epochs 3.0 \
			       --step1_output_dir $Step1_Output_DIR \
			        --step2_output_dir $Step2_Output_DIR \
			        --output_dir $Step2_Model_DIR/$TASK_NAME/
