import alphabets

# about data and net
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzđ: ượê#-.ỉ+~ớâấýạ`íàệẹảộơứũốăòằữờáôắỗểìậổựềởóủẳửầụ,ừồõùếọĩễị'%ẽúãẩéỏặẫỷèỳẻ$ẵỡỹ@_;ỵ"
keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiemnt
random_sample = True # whether to sample the dataset with random sampler
imgH = 32 # the height of the input image to network
imgW = 320 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 1
pretrained = '' # path to pretrained model (to continue training)
expr_dir = 'expr' # where to store samples and models
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

nclass = 1
