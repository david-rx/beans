import pandas as pd

def get_esc_labels():
    target_dir = 'data/esc50'
    df = pd.read_csv(f'{target_dir}/meta/esc50.csv')

    label_dict = {}
    for index, row in df.iterrows():
        label_index = int(row['target'])
        label = row["category"]
        label_dict[label_index] = label


    for i in range(50):
        print(label_dict[i])



def get_cbi_labels():
    label_dict = {}
    curr_labels = ["aldfly", "ameavo", "amebit", "amecro", "amegfi", "amekes", "amepip", "amered", "amerob", "amewig", "amewoo", "amtspa", "annhum", "astfly", "baisan", "baleag", "balori", "banswa", "barswa", "bawwar", "belkin1", "belspa2", "bewwre", "bkbcuc", "bkbmag1", "bkbwar", "bkcchi", "bkchum", "bkhgro", "bkpwar", "bktspa", "blkpho", "blugrb1", "blujay", "bnhcow", "boboli", "bongul", "brdowl", "brebla", "brespa", "brncre", "brnthr", "brthum", "brwhaw", "btbwar", "btnwar", "btywar", "buffle", "buggna", "buhvir", "bulori", "bushti", "buwtea", "buwwar", "cacwre", "calgul", "calqua", "camwar", "cangoo", "canwar", "canwre", "carwre", "casfin", "caster1", "casvir", "cedwax", "chispa", "chiswi", "chswar", "chukar", "clanut", "cliswa", "comgol", "comgra", "comloo", "commer", "comnig", "comrav", "comred", "comter", "comyel", "coohaw", "coshum", "cowscj1", "daejun", "doccor", "dowwoo", "dusfly", "eargre", "easblu", "easkin", "easmea", "easpho", "eastow", "eawpew", "eucdov", "eursta", "evegro", "fiespa", "fiscro", "foxspa", "gadwal", "gcrfin", "gnttow", "gnwtea", "gockin", "gocspa", "goleag", "grbher3", "grcfly", "greegr", "greroa", "greyel", "grhowl", "grnher", "grtgra", "grycat", "gryfly", "haiwoo", "hamfly", "hergul", "herthr", "hoomer", "hoowar", "horgre", "horlar", "houfin", "houspa", "houwre", "indbun", "juntit1", "killde", "labwoo", "larspa", "lazbun", "leabit", "leafly", "leasan", "lecthr", "lesgol", "lesnig", "lesyel", "lewwoo", "linspa", "lobcur", "lobdow", "logshr", "lotduc", "louwat", "macwar", "magwar", "mallar3", "marwre", "merlin", "moublu", "mouchi", "moudov", "norcar", "norfli", "norhar2", "normoc", "norpar", "norpin", "norsho", "norwat", "nrwswa", "nutwoo", "olsfly", "orcwar", "osprey", "ovenbi1", "palwar", "pasfly", "pecsan", "perfal", "phaino", "pibgre", "pilwoo", "pingro", "pinjay", "pinsis", "pinwar", "plsvir", "prawar", "purfin", "pygnut", "rebmer", "rebnut", "rebsap", "rebwoo", "redcro", "redhea", "reevir1", "renpha", "reshaw", "rethaw", "rewbla", "ribgul", "rinduc", "robgro", "rocpig", "rocwre", "rthhum", "ruckin", "rudduc", "rufgro", "rufhum", "rusbla", "sagspa1", "sagthr", "savspa", "saypho", "scatan", "scoori", "semplo", "semsan", "sheowl", "shshaw", "snobun", "snogoo", "solsan", "sonspa", "sora", "sposan", "spotow", "stejay", "swahaw", "swaspa", "swathr", "treswa", "truswa", "tuftit", "tunswa", "veery", "vesspa", "vigswa", "warvir", "wesblu", "wesgre", "weskin", "wesmea", "wessan", "westan", "wewpew", "whbnut", "whcspa", "whfibi", "whtspa", "whtswi", "wilfly", "wilsni1", "wiltur", "winwre3", "wlswar", "wooduc", "wooscj2", "woothr", "y00475", "yebfly", "yebsap", "yehbla", "yelwar", "yerwar", "yetvir"]


    df = pd.read_csv('data/cbi/train.csv')

    for index, row in df.iterrows():
        bird_code = row["ebird_code"]
        label = row["species"]
        label_dict[bird_code] = label

    for key in curr_labels:
        print(label_dict[key])

if __name__ == "__main__":
    get_cbi_labels()





