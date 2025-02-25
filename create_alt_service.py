from pathlib import Path
from classes.msa import MSA


def create_multiple_msa_alt(msa_file_path: Path, msa_name: str, output_dir_path: Path):
    msa = MSA(msa_name)
    msa.read_me_from_fasta(msa_file_path)
    alternatives: list[list[str]] = msa.create_alternative_msas_by_moving_one_part()
    for i, m in enumerate(alternatives):
        inf_alt_msa = MSA(f'{msa_name}_alt_{i}')
        inf_alt_msa.set_sequences_to_me(m, msa.seq_names)
        inf_alt_msa.print_me_to_fasta_file(output_dir_path)
    print('done')


create_multiple_msa_alt('C:/Users/Nimrod.Serok/Nimrod/PhDB/sp_alt/code/sp_alternative/comparison_files/AATF/MSA.MAFFT.aln.With_Names',
                        'MSA.MAFFT.aln.With_Names',
                        'C:/Users/Nimrod.Serok/Nimrod/PhDB/sp_alt/code/sp_alternative/alt_out')



