import re

PAPER_MAPPING = { "/annunci/cerca/$": "search",
        "/annunci/cerca/\\?": "search",
        "/annunci/vendita/\\?page=([0-9])+": "sales_page, page_" + "gr(1)",
        "/annunci/vendita/\\?fb_xd_fragment": "sales_page, facebook",
        "/annunci/vendita(/)?$": "sales_page, page_1",
        "/annunci/vendita/(\\w+)": "sales_anncs",
        "/annunci/affitto(/)?$": "renting_page, page_1",
        "/annunci/affitto/\\?page=([0-9])+": "renting_page, page_" + "gr(1)",
        "/annunci/affitto/(\\w+)": "renting_anncs",
        "^/$": "homepage",
        "^http://www.findyourhouse.it/$": "homepage",
        "/agenzia/": "agency",
        "/news/list/": "news_page",
        "/news/(\\d+)/": "news_article",
        "/contattaci/info/success/": "contacts_requested",
        "/contattaci/privacy/": "tou",
        "/contattaci/ajax/info/$": "contact_requested",
        "/admin/":"admin",
        "/contattaci/info/$" : "contact_requested",
        '/contattaci/valutazione/$': "contact_requested",
    }

class URLMapper:

    def __init__(self):
        self.compiled_regexp = {}
        for reg in PAPER_MAPPING:
            self.compiled_regexp[re.compile(reg)] = PAPER_MAPPING[reg]

    def get_url_label(self, url, req_type, raw_line):
        for regexp in self.compiled_regexp:
            matched = regexp.match(url)
            if matched:
                label = self.compiled_regexp[regexp]
                if "gr(1)" in label:  ## HANDLE LABELS WITH NUMBERS
                    val = matched.group(1)
                    label = label.replace("gr(1)", val)

                ## HANDLE ADMIN PAGES
                if regexp.pattern == '/admin/':
                    if req_type.strip().upper() == "POST":
                        if " 302 " in raw_line:
                            label = "login_success"
                        else:
                            label = "login_fail"

                ## HANDLE CONTACT PAGES
                if regexp.pattern == '/contattaci/info/$':
                    if req_type.strip().upper() != "POST":
                        label = "contacts"
                if regexp.pattern == '/contattaci/valutazione/$':
                    if req_type.strip().upper() != "POST":
                        label = "contacts"
                return label # only match url to a single label