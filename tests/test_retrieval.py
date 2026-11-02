from btu_rag.retriever import Filters, Retriever, parse_request, tokenize


def test_decodes_compact_catalogue_rows(catalogue):
    m = catalogue.by_id["11101"]
    assert m.title_en == "Linear Algebra and Analytical Geometry I"
    assert m.title_de == "Lineare Algebra und analytische Geometrie I"
    assert (m.credits, m.language, m.semester_group) == (8, "German", "winter")
    assert "MAP" in m.exam and m.active
    assert m.url == "https://btu-mofi.com/modules/11101/"


def test_english_first_modules_keep_titles_the_right_way_round(catalogue):
    m = catalogue.by_id["11889"]
    assert (m.title_en, m.language) == ("Introduction to Cyber Security", "English")
    assert m.title_de == "Einführung in die IT-Sicherheit"


def test_tokenizer_keeps_roman_numeral_but_drops_pronoun():
    assert tokenize("Linear Algebra I") == ["linear", "algebra", "i"]
    assert "i" not in tokenize("Can I take statistics?")
    # Umlauts fold and plural/inflection endings are stripped, so both spellings meet in the index.
    assert tokenize("Künstliche Intelligenz") == tokenize("kunstlichen intelligenz") == ["kunstlich", "intelligenz"]


def test_module_number_is_an_exact_hit(catalogue):
    hits = Retriever(catalogue).search("what is 11847", k=3)
    assert hits[0].module.id == "11847" and hits[0].exact_id


def test_german_title_finds_the_module(catalogue):
    hits = Retriever(catalogue).search("Betriebssysteme I", k=3)
    assert hits[0].module.id == "12204"


def test_numeral_separates_course_parts(catalogue):
    r = Retriever(catalogue)
    assert r.search("linear algebra I")[0].module.id == "11101"
    assert r.search("linear algebra II")[0].module.id == "11102"


def test_filters_restrict_results(catalogue):
    hits = Retriever(catalogue).search("cyber security", Filters(level="master", language="English"), k=10)
    assert hits and all(h.module.language == "English" and "master" in h.module.levels for h in hits)


def test_filter_only_request_lists_programme_modules(catalogue):
    query, filters = parse_request("Which modules are in the Artificial Intelligence programme?",
                                   catalogue.programme_names)
    assert filters.programme == "Artificial Intelligence"
    hits = Retriever(catalogue).search(query, filters, k=5)
    assert len(hits) == 5 and all("Artificial Intelligence" in h.module.programmes for h in hits)


def test_no_terms_and_no_filters_returns_nothing(catalogue):
    assert Retriever(catalogue).search("what is the", k=5) == []


def test_parse_request_reads_level_language_and_semester():
    query, filters = parse_request("English-taught master modules on robotics in the summer")
    assert filters == Filters(level="master", language="English", semester="summer")
    assert "robotics" in query and "master" not in query
