FILTERING_TERMS = ["/media/", "/favicon.ico", "/captcha/image", "/fancybox/", "ajax", ".asp", \
                        "/?ag=&r=&pr=&tv=&p=&l", "robots.txt", "msnbot", "Googlebot", "Ezooms", \
                        "AhrefsBot", "TurnitinBot", "Java", "crawler" ]


def is_filtered_url(url):

    if url.endswith("\"-\" \"-\""):
        return True
    for x in FILTERING_TERMS:
        if x in url:
            return True
    return False


