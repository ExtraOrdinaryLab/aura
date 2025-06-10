import os
import re
import csv
import json
import gzip
import shutil
import pathlib
import argparse
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict

import pandas as pd
import soundfile as sf
from tqdm import tqdm
from rich.console import Console

console = Console()
SAMPLERATE = 8000


def download_transcripts(target_folder):
    """
    Download and unpack Switchboard transcripts from OpenSLR.

    Arguments
    ---------
    target_folder : str
        Desired location to store the transcripts.
    """
    transcription_dir = os.path.join(target_folder, "swb_ms98_transcriptions")

    if not os.path.exists(transcription_dir):
        console.log(
            f"Download transcriptions and store them in {target_folder}"
        )

        download_source = "http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz"
        download_target = os.path.join(
            target_folder, "switchboard_word_alignments.tar.gz"
        )
        download_file(download_source, download_target, unpack=True)
    else:
        console.log(
            f"Skipping download of transcriptions because {target_folder} already exists."
        )


def download_file(
    source,
    dest,
    unpack=False,
    dest_unpack=None,
    replace_existing=False,
    write_permissions=False,
):
    """Downloads the file from the given source and saves it in the given
    destination path.

     Arguments
    ---------
    source : path or url
        Path of the source file. If the source is an URL, it downloads it from
        the web.
    dest : path
        Destination path.
    unpack : bool
        If True, it unpacks the data in the dest folder.
        The archive is preserved.

        File formats supported for unpacking/decompression are:

        - any format enumerated by `shutil.get_archive_formats()`, usually
          including `.tar`, `.tar.gz`, `.zip`.
        - plain `.gz` file (when not a `.tar` archive)

        Note that you should ALWAYS trust an archive you are extracting, for
        security reasons.
    dest_unpack: path
        Path where to store the unpacked dataset
    replace_existing : bool
        If True, replaces the existing files.
    write_permissions: bool
        When set to True, all the files in the dest_unpack directory will be granted write permissions.
        This option is active only when unpack=True.
    """
    class DownloadProgressBar(tqdm):
        """DownloadProgressBar class."""

        def update_to(self, b=1, bsize=1, tsize=None):
            """Needed to support multigpu training."""
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    # Create the destination directory if it doesn't exist
    dest_dir = pathlib.Path(dest).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "http" not in source:
        shutil.copyfile(source, dest)

    elif not os.path.isfile(dest) or (
        os.path.isfile(dest) and replace_existing
    ):
        print(f"Downloading {source} to {dest}")
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=source.split("/")[-1],
        ) as t:
            urllib.request.urlretrieve(
                source, filename=dest, reporthook=t.update_to
            )
    else:
        print(f"{dest} exists. Skipping download")

    # Unpack if necessary
    if unpack:
        if dest_unpack is None:
            dest_unpack = os.path.dirname(dest)
        print(f"Extracting {dest} to {dest_unpack}")

        if dest.endswith(".gz") and not dest.endswith(".tar.gz"):
            # just a gzip'd file, but not an actual archive.
            # merely uncompress it and remove the `.gz`.
            with gzip.open(dest, "rb") as f_in:
                with open(dest[:-3], "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.unpack_archive(dest, dest_unpack)

        if write_permissions:
            set_writing_permissions(dest_unpack)


def set_writing_permissions(folder_path):
    """
    This function sets user writing permissions to all the files in the given folder.

    Arguments
    ---------
    folder_path : folder
        Folder whose files will be granted write permissions.
    """
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Set writing permissions (mode 0o666) to the file
            os.chmod(file_path, 0o666)


def match_swbd1(text):
    """
    Clean transcripts in the Switchboard-1 training data.
    The transformations we do are:
     - remove laughter markings, e.g. [LAUGHTER-STORY] -> STORY
     - Remove partial-words, e.g. -[40]1K becomes -1K and -[AN]Y IY becomes -Y
    Also, curly braces, which appear to be used for "nonstandard"
    words or non-words, are removed, e.g. {WOLMANIZED} -> WOLMANIZED

    This is similar to Kaldi's swbd1_map_words.pl.

    Arguments
    ---------
    text : str
        Input text from the Switchboard-1 training data.

    Returns
    -------
    A string containing the cleaned sentence.
    """
    tokens = text.split()
    parsed_tokens = []
    # cspell:disable
    for token in tokens:
        # e.g. [LAUGHTER-STORY] -> STORY; elem 1 and 3 relate to preserving trailing "-"
        m = re.match(r"(|-)^\[LAUGHTER-(.+)\](|-)$", token, flags=re.IGNORECASE)
        token = "".join(m.group(1, 2, 3)) if m else token

        # e.g. [IT'N/ISN'T] -> IT'N
        # Note: 1st part may include partial-word stuff, which we process further below,
        # e.g. [LEM[GUINI]-/LINGUINI]
        m = re.match(r"^\[(.+)/.+\](|-)$", token)
        token = "".join(m.group(1, 2)) if m else token

        # e.g. -[AN]Y -> -Y
        m = re.match(r"^(|-)\[[^][]+\](.+)$", token)
        token = "-" + m.group(2) if m else token

        # e.g. AB[SOLUTE]- -> AB-;
        m = re.match(r"^(.+)\[[^][]+\](|-)$", token)
        token = "".join(m.group(1, 2)) if m else token

        # e.g. EX[SPECIALLY]-/ESPECIALLY] -> EX
        m = re.match(r"([^][]+)\[.+\]$", token)
        token = m.group(1) if m else token

        # e.g. {YUPPIEDOM} -> YUPPIEDOM
        m = re.match(r"^\{(.+)\}$", token)
        token = m.group(1) if m else token

        # e.g. AMMU[N]IT- -> AMMU-IT
        m = re.match(r"(\w+)\[([^][])+\](\w+)", token)
        token = m.group(1) + "-" + m.group(3) if m else token

        # e.g. THEM_1 -> THEM
        token = re.sub(r"_\d+$", "", token)
        parsed_tokens.append(token)
    return " ".join(parsed_tokens)
    # cspell:enable


def match_eval2000(text):
    """
    Clean transcripts in the 2000 Hub5 english evaluation test (LDC2002S09  LDC2002T43)
    See:
    http://www.ldc.upenn.edu/Catalog/catalogEntry.jsp?catalogId=LDC2002S09
    http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC2002T43

    This is similar to eval2000_data_prep.sh

    Arguments
    ---------
    text : str
        Input text from the eval2000 test data.

    Returns
    -------
    A string containing the cleaned sentence.
    """
    cleaned_text = ""

    # Remove utterance when it's just optional nonwords
    text = text.strip().upper()
    for nw in ["UM-HUM", "UMM", "UH-HUH", "MHM", "UH-OH"]:
        text = text.replace(nw, "")

    if "IGNORE_TIME_SEGMENT_" not in text:
        # Remove <B_ASIDE> and <E_ASIDE>.
        cleaned_text = re.sub(r"<.*?>", "", text)
        # Remove everything that is declared optional e.g. (%HESITATION) or (WE-)
        cleaned_text = re.sub(r"[\(\[].*?[\)\]]", "", cleaned_text)
    else:
        console.log(f"Ignoring eval2000 segment: {text}")

    return cleaned_text


def match_fisher(text):
    """
    Clean transcripts in the Fisher corpus.

    This is similar to fisher_data_prep.sh

    Arguments
    ---------
    text : str
        Input text from the Fisher data.

    Returns
    -------
    A string containing the cleaned sentence.
    """

    cleaned_text = ""

    # Remove utterance when it's just optional nonwords
    text = text.strip().upper()
    for nw in ["UM-HUM", "UMM", "UH-HUH", "MHM", "UH-OH"]:
        text = text.replace(nw, "")

    if "((" not in text:
        cleaned_text = re.sub(
            r"\[laugh\]", "[laughter]", text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(
            r"\[sigh\]", "[noise]", cleaned_text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(
            r"\[cough\]", "[noise]", cleaned_text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(
            r"\[sigh\]", "[noise]", cleaned_text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(
            r"\[mn\]", "[noise]", cleaned_text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(
            r"\[breath\]", "[noise]", cleaned_text, flags=re.IGNORECASE
        )
        cleaned_text = re.sub(
            r"\[lipsmack\]", "[noise]", cleaned_text, flags=re.IGNORECASE
        )
    return cleaned_text


def remove_acronym_symbols(text):
    """
    Remove symbols according to the Fisher acronym convention.
    This splits acronyms written as u._c._l._a._ into single characters (e.g. u c l a)

    Arguments
    ---------
    text : str
        Input text

    Returns
    -------
    A string containing the cleaned text.

    """
    cleaned_text = re.sub(r"\._", " ", text)
    cleaned_text = re.sub(r"\.", "", cleaned_text)
    cleaned_text = re.sub(r"them_1", "them", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text


def map_acronyms(dict_acronym, dict_acronym_noi, transcription):
    """
    Transform acronyms in Switchboard transcripts into Fisher corpus convention.

    Examples we want to convert:
    IBM to i._b._m.
    BBC to b._b._c.
    BBCs to b._b._c.s

    This is what Kaldi's map_acronyms_transcripts.py does.

    Arguments
    ---------
    dict_acronym : dict
        Mapping from swbd acronyms to acronyms according to the Fisher corpus convention
    dict_acronym_noi : dict
        Mapping from swbd acronyms to acronyms according to the Fisher corpus convention with the letter I removed
    transcription : str
        A sentence in the Switchboard transcripts
    Returns
    -------
    The original sentence but with acronyms according to the Fisher convention
    """

    items = transcription.split()
    utt_length = len(items)
    # First pass mapping to map I as part of acronym
    for i in range(utt_length):
        if items[i] == "I":
            x = 0
            while i - 1 - x >= 0 and re.match(r"^[A-Z]$", items[i - 1 - x]):
                x += 1

            y = 0
            while i + 1 + y < utt_length and re.match(
                r"^[A-Z]$", items[i + 1 + y]
            ):
                y += 1

            if x + y > 0:
                for bias in range(-x, y + 1):
                    items[i + bias] = dict_acronym[items[i + bias]]

    # Second pass mapping (not mapping 'i' and 'I')
    for i in range(len(items)):
        if items[i] in dict_acronym_noi.keys():
            items[i] = dict_acronym_noi[items[i]]
    sentence = " ".join(items[1:])

    return items[0] + " " + sentence


def make_name_to_disk_dict(mapping_table: str):
    """
    The Switchboard data is spread across 4 DVDs
    represented by directories ("swb1_d1", "swb1_d2" and so on).
    This function creates a lookup dictionary to map a given filename to the
    disk it was stored on.
    This information is useful to assemble the absolute path to the sph audio
    files.

    Arguments
    ---------
    mapping_table : str
        String representing the path to the mapping table file "swb1_all.dvd.tbl"
        provided along with the rest of the Switchboard data.

    Returns
    -------
    name2disk : dict
        A dictionary that maps from sph filename (key) to disk-id (value)
    """
    name2disk = {}
    with open(mapping_table, encoding="utf-8") as mt:
        for line in mt:
            split = line.split()
            name = split[1].strip()
            name2disk[name] = split[0].strip()
    return name2disk


def filter_text(
    transcription: str, dataset="train", acronyms=None, acronyms_noi=None
):
    """
    This function takes a string representing a sentence in one
    of the datasets and cleans it using various regular expressions.
    The types of regular expressions applied depend on the dataset.

    Arguments
    ---------
    transcription : str
        A transcribed sentence
    dataset : str
        Either "train", "eval2000", or "fisher" depending on the type
        of data you want to clean.
    acronyms : dict
        Dictionary mapping acronyms to Fisher convention (only relevant for swbd1 training data)
    acronyms_noi : dict
        Dictionary mapping acronyms to Fisher convention without I (only relevant for swbd1 training data)

    Returns
    -------
    A string containing the cleaned sentence.

    """
    dataset = dataset.strip().lower()

    if dataset == "train":
        # This is similar to what swbd1_data_prep.sh and swbd1_map_words.pl does.
        transcription = re.sub(
            r"\[SILENCE\]", "", transcription, flags=re.IGNORECASE
        )
        transcription = re.sub(r"<.*?>", "", transcription)
        transcription = match_swbd1(transcription.strip())

        transcription = re.sub(r"\s\s+", " ", transcription)

        # Convert acronyms to Fisher convention
        if len(transcription) > 0:
            transcription = map_acronyms(acronyms, acronyms_noi, transcription)

        # Split acronyms written as u._c._l._a._ into single characters (e.g. u c l a)
        transcription = remove_acronym_symbols(transcription)
        transcription = transcription.upper().strip()

    elif dataset in ["eval2000", "hub5", "test"]:
        # This is similar to what eval2000_data_prep.sh does.
        transcription = match_eval2000(transcription.strip())
    elif dataset == "fisher":
        # This is similar to what fisher_data_prep.sh does.
        transcription = match_fisher(transcription.strip())
    else:
        raise NameError(f"Invalid dataset descriptor '{dataset}' supplied.")

    # Remove redundant whitespaces
    transcription = re.sub(r"\s\s+", " ", transcription)
    return transcription.strip()


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Returns
    -------
    allFiles : list
        The list of files matching the patterns.

    Example
    -------
    >>> get_all_files('tests/samples/RIRs', match_and=['3.wav'])
    ['tests/samples/RIRs/rir3.wav']
    """
    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles


def check_data_folder(root_folder):
    """
    Check if all directories exist to prepare the Switchboard dataset.

    Arguments
    ---------
    root_folder : str
        Root directory, where the Switchboard data is located.
        Expects the following subdirectories to exist:
        "docs", "swb1_d1", "swb1_d2", "swb1_d3", "swb1_d4"
    """
    for sub_folder in ["docs", "swb1_d1", "swb1_d2", "swb1_d3", "swb1_d4"]:
        swbd_folder = os.path.join(root_folder, sub_folder)
        if not os.path.exists(swbd_folder):
            err_msg = f"The folder {swbd_folder} does not exist (it is expected in the Switchboard dataset)"
            raise OSError(err_msg)


def prepare_lexicon(lexicon_file, output_file):
    """
    Prepare the swbd1 lexicon for further processing.
    The lexicon is used to find acronyms and to convert them into Fisher convention.

    Arguments
    ---------
    lexicon_file : str
        Path to the sw-ms98-dict.text file in the Switchboard corpus
    output_file : str
        Path to store the cleaned lexicon at

    Returns
    -------
    A list containing the cleaned lexicon entries

    """
    lexicon = []
    lex_out = open(output_file, "w", encoding="utf-8")
    with open(lexicon_file, encoding="utf-8") as lf:
        # Skip first row
        next(lf)
        for line in lf:
            # Skip header
            if line.startswith("#"):
                continue
            parsed_line = match_swbd1(line.strip())
            if len(parsed_line) > 0:
                lexicon.append(parsed_line)
                lex_out.write(f"{parsed_line}\n")
    return lexicon


def make_acronym_map(save_folder, lexicon_file, acronym_map_file):
    """
    Create mappings that can be used to convert acronyms in the Switchboard corpus
    into acronyms using the Fisher corpus convention.

    Examples we want to convert:
    IBM to i._b._m.
    BBC to b._b._c.
    BBCs to b._b._c.s

    This is what Kaldi's format_acronyms_dict.py does.

    Arguments
    ---------
    save_folder : str
        Folder to store the acronym map on disk
    lexicon_file : str
        Path to the sw-ms98-dict.text file
    acronym_map_file : str
        File to store the acronym map in

    Returns
    -------
    Two dictionaries mapping from swbd acronyms to acronyms according to the Fisher corpus convention.
    The first dict contains all entries, the other has the letter I removed.
    """

    # Taken from https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c/local/MSU_single_letter.txt
    msu_single_letter = [
        "A ey",
        "B b iy",
        "C s iy",
        "D d iy",
        "E iy",
        "F eh f",
        "G jh iy",
        "H ey ch",
        "I ay",
        "J jh ey",
        "K k ey",
        "L eh l",
        "M eh m",
        "N eh n",
        "O ow",
        "P p iy",
        "Q k y uw",
        "R aa r",
        "S eh s",
        "T t iy",
        "U y uw",
        "V v iy",
        "W d ah b ax l y uw",
        "X eh k s",
        "Y w ay",
        "Z z iy",
    ]

    fin_lex = (
        prepare_lexicon(lexicon_file, os.path.join(save_folder, "lexicon.txt"))
        + msu_single_letter
    )
    console.log(
        f"Prepared Swbd1 + MSU single letter lexicon with {len(fin_lex)} entries"
    )
    fout_map = open(acronym_map_file, "w", encoding="utf-8")

    # Initialise single letter dictionary
    dict_letter = {}
    for single_letter_lex in msu_single_letter:
        items = single_letter_lex.split()
        dict_letter[items[0]] = single_letter_lex[len(items[0]) + 1 :].strip()

    for lex in fin_lex:
        items = lex.split()
        word = items[0]
        lexicon = lex[len(items[0]) + 1 :].strip()
        # find acronyms from words with only letters and '
        pre_match = re.match(r"^[A-Za-z]+$|^[A-Za-z]+\'s$|^[A-Za-z]+s$", word)
        if pre_match:
            # find if words in the form of xxx's is acronym
            if word[-2:] == "'s" and (lexicon[-1] == "s" or lexicon[-1] == "z"):
                actual_word = word[:-2]
                actual_lexicon = lexicon[:-2]
                acronym_lexicon = ""
                for w in actual_word:
                    acronym_lexicon = (
                        acronym_lexicon + dict_letter[w.upper()] + " "
                    )
                if acronym_lexicon.strip() == actual_lexicon:
                    acronym_mapped = ""
                    acronym_mapped_back = ""
                    for w in actual_word[:-1]:
                        acronym_mapped = acronym_mapped + w.lower() + "._"
                        acronym_mapped_back = (
                            acronym_mapped_back + w.lower() + " "
                        )
                    acronym_mapped = (
                        acronym_mapped + actual_word[-1].lower() + ".'s"
                    )
                    acronym_mapped_back = (
                        acronym_mapped_back + actual_word[-1].lower() + "'s"
                    )
                    fout_map.write(
                        word
                        + "\t"
                        + acronym_mapped
                        + "\t"
                        + acronym_mapped_back
                        + "\n"
                    )

            # find if words in the form of xxxs is acronym # cspell:ignore xxxs
            elif word[-1] == "s" and (lexicon[-1] == "s" or lexicon[-1] == "z"):
                actual_word = word[:-1]
                actual_lexicon = lexicon[:-2]
                acronym_lexicon = ""
                for w in actual_word:
                    acronym_lexicon = (
                        acronym_lexicon + dict_letter[w.upper()] + " "
                    )
                if acronym_lexicon.strip() == actual_lexicon:
                    acronym_mapped = ""
                    acronym_mapped_back = ""
                    for w in actual_word[:-1]:
                        acronym_mapped = acronym_mapped + w.lower() + "._"
                        acronym_mapped_back = (
                            acronym_mapped_back + w.lower() + " "
                        )
                    acronym_mapped = (
                        acronym_mapped + actual_word[-1].lower() + ".s"
                    )
                    acronym_mapped_back = (
                        acronym_mapped_back + actual_word[-1].lower() + "'s"
                    )
                    fout_map.write(
                        word
                        + "\t"
                        + acronym_mapped
                        + "\t"
                        + acronym_mapped_back
                        + "\n"
                    )

            # find if words in the form of xxx (not ended with 's or s) is acronym
            elif word.find("'") == -1 and word[-1] != "s":
                acronym_lexicon = ""
                for w in word:
                    acronym_lexicon = (
                        acronym_lexicon + dict_letter[w.upper()] + " "
                    )
                if acronym_lexicon.strip() == lexicon:
                    acronym_mapped = ""
                    acronym_mapped_back = ""
                    for w in word[:-1]:
                        acronym_mapped = acronym_mapped + w.lower() + "._"
                        acronym_mapped_back = (
                            acronym_mapped_back + w.lower() + " "
                        )
                    acronym_mapped = acronym_mapped + word[-1].lower() + "."
                    acronym_mapped_back = acronym_mapped_back + word[-1].lower()
                    fout_map.write(
                        word
                        + "\t"
                        + acronym_mapped
                        + "\t"
                        + acronym_mapped_back
                        + "\n"
                    )

    fout_map.close()

    # Load acronym map for further processing
    fin_map = open(acronym_map_file, "r", encoding="utf-8")
    dict_acronym = {}
    dict_acronym_noi = {}  # Mapping of acronyms without I, i
    for pair in fin_map:
        items = pair.split("\t")
        dict_acronym[items[0]] = items[1]
        dict_acronym_noi[items[0]] = items[1]
    fin_map.close()
    del dict_acronym_noi["I"]
    del dict_acronym_noi["i"]

    return dict_acronym, dict_acronym_noi


def skip(*filenames):
    """
    Detects if the Switchboard data preparation has already been done.

    Arguments
    ---------
    *filenames : tuple
        List of paths to check for existence.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, preparation must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def merge_csvs(data_folder, csv_lst, merged_csv):
    """Merging several csv files into one file.

    Arguments
    ---------
    data_folder : string
        The folder to store csv files to be merged and after merging.
    csv_lst : list
        Filenames of csv file to be merged.
    merged_csv : string
        The filename to write the merged csv file.

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> os.symlink(os.path.realpath("tests/samples/annotation/speech.csv"), tmpdir / "speech.csv")
    >>> merge_csvs(tmpdir,
    ... ["speech.csv", "speech.csv"],
    ... "test_csv_merge.csv")
    """
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        console.log("Skipping merging. Completed in previous run.")
    with open(
        os.path.join(data_folder, csv_lst[0]), newline="", encoding="utf-8"
    ) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_lst:
        with open(
            os.path.join(data_folder, csv_file), newline="", encoding="utf-8"
        ) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError(
                            "Different header for " f"{csv_lst[0]} and {csv}."
                        )
                    continue
                lines.append(line)
    with open(write_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    console.log(f"{write_path} is created.")


def maybe_merge_files(merge_name, merge_lst: list, save_folder):
    """

    Merge multiple .csv files and store the combined data in a new file.

    Arguments
    ---------
    merge_name : str
        New name to save the combined files under.
    merge_lst  : list
        List of data splits to be merged.

    """
    if len(merge_lst) > 1:
        if merge_name is not None and len(merge_name) > 0:
            merge_files = [data_split + ".csv" for data_split in merge_lst]
            merge_csvs(
                data_folder=save_folder,
                csv_lst=merge_files,
                merged_csv=merge_name,
            )
        else:
            console.log(
                "No name for merged .csv supplied. "
                "You can pass a name for the merged .csv files "
                "via the merge_name parameter. Not combining any .csv files!"
            )


def swbd1_data_prep(
    data_folder,
    save_folder,
    add_fisher_corpus=False,
    max_utt=9999999999,
):
    """
    Prepare the Switchboard Phase 1 training data (LDC97S62).
    Only creates a single split: 'train.csv'
    """
    console.log("Starting data preparation for main Switchboard corpus")

    train_data_folder = os.path.join(data_folder, "LDC97S62")
    check_data_folder(train_data_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    download_transcripts(save_folder)

    # Make a mapping from Switchboard acronyms to Fisher convention
    console.log("Preparing acronym mappings")
    lexicon_input_file = os.path.join(
        save_folder, "swb_ms98_transcriptions", "sw-ms98-dict.text"
    )
    acronym_map_output_file = os.path.join(save_folder, "acronyms.map")
    dict_acronym, dict_acronym_noi = make_acronym_map(
        save_folder, lexicon_input_file, acronym_map_output_file
    )

    # collect all files containing transcriptions
    transcript_files = get_all_files(
        os.path.join(save_folder, "swb_ms98_transcriptions"),
        match_and=["trans.text"],
    )

    name2disk = make_name_to_disk_dict(
        os.path.join(train_data_folder, "docs/swb1_all.dvd.tbl")
    )
    console.log(
        f"Made name2disk mapping dict containing {len(name2disk)} conversations."
    )

    csv_lines = [
        [
            "ID",
            "duration",
            "start",
            "stop",
            "channel",
            "wav",
            "words",
            "spk_id",
        ]
    ]
    swbd_train_lines = []
    for filename in transcript_files:
        with open(filename, encoding="utf-8") as file:
            for line in file:
                str_split = line.split()
                id = str_split[0].strip()
                channel = id.split("-")[0][-1]
                wav_name = id.split("-")[0][:6] + ".sph"
                spk_id = wav_name.replace(".sph", channel)
                wav_name = wav_name.replace(wav_name[0:2], "sw0")
                disk = name2disk[wav_name]

                wav_path = os.path.join(
                    train_data_folder, disk, "data", wav_name
                )
                seg_start = int(float(str_split[1].strip()) * SAMPLERATE)
                seg_end = int(float(str_split[2].strip()) * SAMPLERATE)
                audio_duration = (seg_end - seg_start) / SAMPLERATE

                transcription = " ".join(str_split[3:])
                cleaned_transcription = filter_text(
                    transcription,
                    dataset="train",
                    acronyms=dict_acronym,
                    acronyms_noi=dict_acronym_noi,
                )

                if len(cleaned_transcription) > 0:
                    csv_lines.append(
                        [
                            id,
                            audio_duration,
                            seg_start,
                            seg_end,
                            channel,
                            wav_path,
                            cleaned_transcription,
                            spk_id,
                        ]
                    )
                    if add_fisher_corpus:
                        swbd_train_lines.append([id, cleaned_transcription])

    csv_file = os.path.join(save_folder, "train.csv")
    console.log(f"Creating csv file {csv_file}")
    write_csv(csv_file, csv_lines, utt_id_idx=6, max_utt=max_utt)
    return swbd_train_lines


def prepare_switchboard(
    data_folder,
    save_folder,
    skip_prep=False,
    add_fisher_corpus=False,
    max_utt=300,
):
    """
    Main function for Switchboard data preparation.
    Only prepares a single split: all Switchboard data as 'train.csv'.
    """
    if skip_prep:
        console.log("Data preparation skipped manually via hparams")
        return

    filenames = [
        os.path.join(save_folder, "train.csv"),
        os.path.join(save_folder, "test.csv"),
    ]
    if add_fisher_corpus:
        filenames.append(os.path.join(save_folder, "train_lm.csv"))

    if skip(*filenames):
        console.log("Preparation completed in previous run, skipping.")
        return

    swbd_train_lines = swbd1_data_prep(
        data_folder,
        save_folder,
        add_fisher_corpus=add_fisher_corpus,
        max_utt=max_utt,
    )

    # Prepare eval2000 testset
    eval2000_data_prep(data_folder, save_folder)

    if add_fisher_corpus:
        fisher_lines = fisher_data_prep(data_folder, save_folder)
        combined_lines = fisher_lines + swbd_train_lines
        csv_file = os.path.join(save_folder, "train_lm.csv")
        write_csv(csv_file, combined_lines, utt_id_idx=1, max_utt=999999999)

    console.log("Switchboard data preparation finished.")


def write_csv(csv_file, csv_lines, utt_id_idx=0, max_utt=300):
    """
    Write utterances to a .csv file.

    Arguments
    ---------
    csv_file : str
        Full path of the file to save
    csv_lines : list
        A list of lists containing the data to write to the .csv file.
    utt_id_idx : int
        Element in the list representing a line that marks the utterance id.
        This is necessary to keep track of duplicate utterances.
    max_utt : int
        Maximum number of duplicate utterances to be written.
        Once max_utt is exceeded, any lines containing the same
        utterance will not be written to the .csv file
    """

    # Keep track of the number of times each utterance appears
    utt2count = defaultdict(int)

    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            current_utt = line[utt_id_idx]
            # Avoid that the same utterance becomes part of the dataset too often
            if utt2count[current_utt] < max_utt:
                csv_writer.writerow(line)

            utt2count[current_utt] += 1


def eval2000_data_prep(data_folder: str, save_folder: str):
    """
    Prepare the eval2000/Hub5 English data (LDC2002S09 and LDC2002T43).
    The data serves as the test set and is separated into
    the full dataset (test.csv), the Switchboard portion
    of the dataset (test_swbd.csv) and the Callhome portion
    of the dataset (test_callhome.csv).

    Arguments
    ---------
    data_folder : str
        Path to the folder where the eval2000/Hub5 English data is located.
    save_folder : str
        The directory to store the csv files at.
    """

    console.log(
        "Begin preparing the eval2000 Hub5 English test set and transcripts (LDC2002S09 and LDC2002T43)"
    )

    audio_folder = os.path.join(data_folder, "LDC2002S09/hub5e_00/english")
    transcription_file = os.path.join(
        data_folder,
        "LDC2002T43/2000_hub5_eng_eval_tr/reference/hub5e00.english.000405.stm",
    )

    for d in [audio_folder, transcription_file]:
        if not os.path.exists(d):
            err_msg = f"The folder {d} does not exist (it is expected to prepare the eval2000/hub5 test set)"
            raise OSError(err_msg)

    csv_lines_callhome = [
        ["ID", "duration", "start", "stop", "channel", "wav", "words", "spk_id"]
    ]
    csv_lines_swbd = [
        ["ID", "duration", "start", "stop", "channel", "wav", "words", "spk_id"]
    ]

    with open(transcription_file, encoding="utf-8") as file:
        utt_count = 0
        for line in file:
            # Skip header
            if line.startswith(";;"):
                continue

            str_split = line.split()
            # Sometimes the end time of a segment is shifted to the right
            # So we remove all empty strings from the split
            str_split = [i for i in str_split if len(i) > 0]

            # Make ID unique
            id = str_split[2].strip() + "_" + str(utt_count)
            channel = str_split[1].strip()

            wav_name = str_split[0].strip() + ".sph"
            wav_path = os.path.join(audio_folder, wav_name)

            spk_id = str_split[2].strip()

            # The label "en" stands for "Callhome conversations"
            # The label "sw" stands for "Switchboard conversations"
            is_swbd = str_split[0].strip().startswith("sw_")

            # We want the segment start and end times in samples,
            # so we can slice the segment from the tensor
            try:
                seg_start = int(float(str_split[3].strip()) * SAMPLERATE)
                seg_end = int(float(str_split[4].strip()) * SAMPLERATE)
            except ValueError:
                console.log(
                    f"Unable to determine start and end time of segment. "
                    f"This should not happen! Split in "
                    f"question was: {str_split}"
                )

            audio_duration = (seg_end - seg_start) / SAMPLERATE

            transcription = " ".join(str_split[6:])
            cleaned_transcription = filter_text(
                transcription, dataset="eval2000"
            )

            # Skip empty transcriptions
            if len(cleaned_transcription) > 0:
                utt_line = [
                    id,
                    audio_duration,
                    seg_start,
                    seg_end,
                    channel,
                    wav_path,
                    cleaned_transcription,
                    spk_id,
                ]
                if is_swbd:
                    csv_lines_swbd.append(utt_line)
                else:
                    csv_lines_callhome.append(utt_line)
            utt_count += 1

    merge_files = []
    for name, lines in [
        ("swbd", csv_lines_swbd),
        ("callhome", csv_lines_callhome),
    ]:
        filename = f"test_{name}.csv"
        csv_file = os.path.join(save_folder, filename)
        console.log(f"Creating csv file {csv_file}")

        with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for line in lines:
                csv_writer.writerow(line)

        merge_files.append(filename)
    merge_csvs(
        data_folder=save_folder, csv_lst=merge_files, merged_csv="test.csv"
    )

    glm_dir = os.path.join(
        data_folder,
        "LDC2002T43/2000_hub5_eng_eval_tr/reference",
    )
    console.log("Start parsing mapping rules in en20000405_hub5.glm")
    parse_glm_file(glm_dir, save_folder)


def parse_glm_file(glm_dir, save_folder):
    """
    Parse the file called en20000405_hub5.glm.
    This file contains the transcript filtering rules for the
    Hub4-E and Hub5-E Evaluations.

    These filtering rules are needed during inference to find valid word alternatives.

    Arguments
    ---------
    glm_dir : str
        Location of the en20000405_hub5.glm file in the eval2000 test set
    save_folder : str
        Directory to store the parsed GLM file
    """
    results = defaultdict(list)
    with open(
        os.path.join(glm_dir, "en20000405_hub5.glm"), encoding="utf-8"
    ) as file:
        is_contraction = False
        for line in file:
            # Skip comments
            if "CONTRACTIONIZER" in line:
                is_contraction = True
            if line.startswith(";;") or line.startswith("*"):
                continue
            line_split = line.split("=>")
            if len(line_split) > 1:
                wrd = line_split[0].replace("[", "").replace("]", "").strip()
                # Split alternative at comment
                if not is_contraction:
                    alternative = line_split[1]
                    alternative = alternative.split(";;")[0].strip()
                    # Split alternative again add additional information
                    alternative = (
                        alternative.split("/")[0]
                        .replace("[", "")
                        .replace("]", "")
                        .strip()
                    )
                    results[wrd] += [alternative]
                else:
                    # Now we parse contraction rules (last 1000 rows or so)
                    alternative = (
                        line_split[1]
                        .replace("/ [ ] __ [ ]", "")
                        .replace("[{", "")
                        .replace("}]", "")
                    )
                    alternatives = alternative.split("/")
                    alternatives = [
                        i.replace("[", "").replace("]", "").strip()
                        for i in alternatives
                    ]
                    results[wrd] += alternatives

    csv_file = os.path.join(save_folder, "glm.csv")
    console.log("Writing GLM csv file")

    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for wrd, alternatives in results.items():
            line = [wrd, "|".join(alternatives)]
            csv_writer.writerow(line)


def fisher_data_prep(data_folder, save_folder):
    """
    Prepare Fisher data for Tokenizer and LM Training.
    The Fisher transcripts are located at
    LDC2004T19/fe_03_p1_tran and LDC2005T19/fe_03_p2_tran.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Fisher data is located.
    save_folder : str
        Path to the folder where you want to store the prepared data.

    Returns
    -------
    A list containing the prepared data for further processing
    """

    console.log(
        "Begin preparing the Fisher corpus transcripts (LDC2002S09 and LDC2002T43)"
    )

    fisher_dirs = [
        "LDC2004T19/fe_03_p1_tran/data/trans",
        "LDC2005T19/fe_03_p2_tran/data/trans",
    ]

    for fisher_dir in fisher_dirs:
        joined_path = os.path.join(data_folder, fisher_dir)
        if not os.path.exists(joined_path):
            err_msg = f"The folder {joined_path} does not exist (it is expected to prepare the Fisher corpus)"
            raise OSError(err_msg)

    csv_lines = [["ID", "words"]]
    num_files_processed = 0
    num_dirs_processed = 0
    utt_count = 0

    for fisher_dir in fisher_dirs:
        joined_path = os.path.join(data_folder, fisher_dir)
        transcript_files = get_all_files(joined_path, match_and=[".txt"])

        for transcript_files in transcript_files:
            with open(transcript_files, encoding="utf-8") as file:
                for line in file:
                    # skip header and empty lines
                    if line.startswith("#") or len(line.strip()) < 1:
                        continue

                    # Create unique id
                    id = "fisher-" + str(utt_count)
                    transcription = line.split()[3:]
                    transcription = " ".join(transcription)
                    transcription_clean = filter_text(
                        transcription, dataset="fisher"
                    )

                    # Split acronyms written as u._c._l._a._ into single characters (e.g. u c l a)
                    transcription_clean = remove_acronym_symbols(
                        transcription_clean
                    )
                    transcription_clean = transcription_clean.upper().strip()

                    # Skip empty transcriptions
                    if len(transcription_clean) > 0:
                        csv_lines.append([id, transcription_clean])
                        utt_count += 1
            # This is just for accounting
            num_files_processed += 1
        num_dirs_processed += 1

    console.log(
        f"Fisher corpus: Processed {num_files_processed} files in "
        f"{num_dirs_processed} directories."
    )

    csv_file = os.path.join(save_folder, "fisher.csv")
    console.log(f"Creating csv file {csv_file}")

    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)
    return csv_lines


def check_sph2pipe_available():
    from shutil import which
    return which('sph2pipe') is not None


def convert_sph_to_wav(sph_path, wav_path):
    if os.path.exists(wav_path):
        return
    cmd = f"sph2pipe -f wav {sph_path} {wav_path}"
    result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sph2pipe convert unsuccessful: {result.stderr}")


def process_csv(csv_path, segment_output_dir, jsonl_path):
    df = pd.read_csv(csv_path)
    os.makedirs(segment_output_dir, exist_ok=True)
    jsonl_lines = []

    # 依照 wav 路徑分組
    df['wav_path'] = df['wav'].apply(lambda x: x.replace('.sph', '.wav'))
    grouped = df.groupby('wav_path')

    for wav_path, group in tqdm(grouped, desc="Process different wav"):
        sph_path = group['wav'].iloc[0]
        try:
            if not os.path.exists(wav_path):
                convert_sph_to_wav(sph_path, wav_path)
        except Exception as e:
            console.log(f"[red]Can not convert {sph_path}: {e}[/red]")
            continue

        # 只開一次 wav
        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            console.log(f"[red]Can not read wav file {wav_path}: {e}[/red]")
            continue

        for _, row in group.iterrows():
            seg_name = f"{row['ID']}.wav"
            seg_path = os.path.join(segment_output_dir, seg_name)
            if os.path.exists(seg_path):
                # 若已存在直接跳過
                pass
            else:
                try:
                    # 擷取 segment
                    start = int(float(row['start']) / 8000 * sr)
                    end = int(float(row['stop']) / 8000 * sr)
                    channel = row['channel']
                    if audio.ndim == 2:
                        channel_idx = 0 if channel == 'A' else 1
                        segment = audio[start:end, channel_idx]
                    else:
                        segment = audio[start:end]
                    sf.write(seg_path, segment, sr)
                except Exception as e:
                    console.log(f"[red]擷取 segment 失敗 {wav_path}: {e}[/red]")
                    continue
            # 組 jsonl
            duration = float(row['duration'])
            json_obj = {
                "id": row['ID'],
                "audio": {"path": seg_path},
                "original_audio": {"path": wav_path},
                "start": float(row['start']) / 8000,
                "end": float(row['stop']) / 8000,
                "duration": duration,
                "channel": row['channel'],
                "sentence": row['words'],
                "sentences": [],
                "spk_id": row['spk_id']
            }
            jsonl_lines.append(json_obj)

    # 輸出 jsonl
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for obj in jsonl_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    console.log(f"[green]完成：{jsonl_path}，共 {len(jsonl_lines)} 條[/green]")


def main():
    parser = argparse.ArgumentParser()
    # LDC97S62, LDC2002T43, LDC2002S09
    parser.add_argument('--data-dir', type=str, required=True, help='')
    parser.add_argument('--output-dir', type=str, required=True, help='')
    args = parser.parse_args()

    prepare_switchboard(
        args.data_dir,
        args.output_dir,
        skip_prep=False,
        add_fisher_corpus=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    seg_dir = os.path.join(args.output_dir, 'segments')
    os.makedirs(seg_dir, exist_ok=True)

    if not check_sph2pipe_available():
        console.log("[red]Please install sph2pipe first[/red]")
        return

    process_csv(
        csv_path=os.path.join(args.output_dir, 'train.csv'),
        segment_output_dir=seg_dir,
        jsonl_path=os.path.join(args.output_dir, 'train.jsonl')
    )
    process_csv(
        csv_path=os.path.join(args.output_dir, 'test.csv'),
        segment_output_dir=seg_dir,
        jsonl_path=os.path.join(args.output_dir, 'test.jsonl')
    )


if __name__ == "__main__":
    main()