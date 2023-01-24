import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import sys
val = sys.argv[1]
# df=pd.read_csv("ds_dataset2.csv")
df=pd.read_csv(val)
pd.set_option('display.max_columns', None)
#dropping certifications and links columns as all values are null
df = df.drop(['Certifications/Achievement/ Research papers', 'Link to updated Resume (Google/ One Drive link preferred)','link to Linkedin profile'] , 1)
# dropping the rows having NaN values 
df = df.dropna() 
df = df.reset_index(drop = True) 

with PdfPages('visualization-output.pdf') as pdf:
    fig = plt.figure()
    df['Areas of interest'].value_counts().plot(kind='barh', figsize=(15, 6))
    plt.ylabel('Areas of Interest')
    plt.xlabel('Number of Applicants')
    plt.title('Q2(a) The number of students applied to different technologies')
    pdf.savefig(fig,bbox_inches = 'tight')
    plt.close()
    
    df2=df[["Areas of interest","Programming Language Known other than Java (one major)"]]
    df2=df2[df2["Areas of interest"]=='Data Science ']
    q2=[]
    q2.append(df2['Programming Language Known other than Java (one major)'].value_counts()['Python'])
    q2.append(len(df2)-q2[0])
    labels = ['Knew Python', "Didn't Know Python"]
    explode = (0.1, 0)
    tot=sum(q2)/100.0
    autopct=lambda x: "%d" % round(x*tot)
    fig2 = plt.figure(figsize = (15,12))
    plt.pie(q2, autopct=autopct, explode=explode, labels=labels, shadow=True)
    plt.title('Q2(b) The number of students applied for Data Science who knew "Python" and who didn’t')
    pdf.savefig(fig2,bbox_inches = 'tight')
    plt.close()
    
    fig3 = plt.figure(figsize = (15,6))
    df['How Did You Hear About This Internship?'].value_counts().plot(kind='bar', figsize=(15, 6))
    plt.ylabel('Number of Applicants')
    plt.xlabel('Ways in which applicants heard about the internship')
    plt.title('Q2(c) The different ways students learned about this program.')
    pdf.savefig(fig3,bbox_inches = 'tight')
    plt.close()
    
    df2d = df[df['Which-year are you studying in?'] == "Fourth-year"]
    less_than8 = df2d[df2d["CGPA/ percentage"] < 8][["CGPA/ percentage"]]["CGPA/ percentage"].count()
    eight_nine = df2d[(df2d["CGPA/ percentage"] >= 8) & (df2d["CGPA/ percentage"] < 9)]["CGPA/ percentage"].count()
    nine_ten = df2d[(df2d["CGPA/ percentage"] >= 9) & (df2d["CGPA/ percentage"] <= 10)]["CGPA/ percentage"].count()
    fig4, ax = plt.subplots()
    # Defining x and y axes
    x = ['Less than 8' , '8 to 9' ,'9 to 10']
    y = [less_than8,eight_nine,nine_ten]
    plt.bar(x, y,0.5)
    plt.title("Q2(d) Students who are in the fourth year and have a CGPA greater than 8.0.")
    plt.xlabel("CGPA")
    plt.ylabel("Number of Students")
    pdf.savefig(fig4,bbox_inches = 'tight')
    plt.close()
    
    df2e = df[df["Areas of interest"]=="Digital Marketing "]
    verbal=df2e[(df2e["Rate your verbal communication skills [1-10]"] > 8)]
    written=df2e[(df2e["Rate your written communication skills [1-10]"] > 8)]
    both=df2e[(df2e["Rate your verbal communication skills [1-10]"] > 8) &  (df2e["Rate your written communication skills [1-10]"] > 8)]
    x = ['Total','Verbal > 8' , 'Written > 8' ,'Both>8']
    y = [len(df2e),len(verbal),len(written),len(both)]
    fig5 = plt.figure(figsize = (15,6))
    plt.bar(x, y,0.5,color='red')
    plt.ylabel('Range of CGPA')
    plt.xlabel('Number Of Students')
    plt.title('Q2(e) Students who applied for Digital Marketing with verbal and written communication score greater than 8.')
    pdf.savefig(fig5,bbox_inches = 'tight')
    plt.close()
    
    lyear=df["Which-year are you studying in?"].value_counts()
    group_names=['First-year', 'Second-year', 'Third-year', 'Fourth-year']
    group_size=[lyear['First-year'],lyear['Second-year'],lyear['Third-year'],lyear['Fourth-year']]
    major_names=list(set(df["Major/Area of Study"]))
    subgroup_size=[]
    subgroup_names=[]
    lab=[]
    k=0
    subgroup_names1=['CSE','EE','EXTC']*4
    label1=[]
    for i in range(len(group_names)):
        for j in range(len(major_names)):
            subgroup_size.append(len(df[(df["Major/Area of Study"]==major_names[j]) & (df["Which-year are you studying in?"]==group_names[i])]))
            subgroup_names.append(major_names[j])
            lab.append(str(subgroup_size[k])+ ' - '+group_names[i] + " - "+ major_names[j])
            label1.append(subgroup_names1[k] + '-'+ str(subgroup_size[k]))
            k+=1
    a, b, c,d=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges]
    # First Ring (outside)
    fig6, ax = plt.subplots(figsize=(15, 6))
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=2.2, labels=group_names, colors=[a(0.6), b(0.6), c(0.6), d(.6)])
    plt.setp( mypie, width=0.7, edgecolor='white')
    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=2.2-0.3,labels=label1,labeldistance=0.8, colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), b(0.3), c(0.5), c(0.4), c(0.3), d(0.5),d(.4),d(.3)])
    plt.setp( mypie2, width=0.5, edgecolor='white')
    plt.margins(0,0)
    plt.legend(loc=(0.9, 0.1))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[4:], lab,loc='center',prop={'size':9 })
    plt.title("Q2(f) Year-wise and area of study wise classification of students.",pad=150)
    pdf.savefig(fig6,bbox_inches = 'tight')
    plt.close()
    
    #City and college wise classification of students.
    fig7 = plt.figure()
    df['City'].value_counts().plot(kind='pie',figsize=(10, 12),autopct='%.2f')
    plt.title('Q2(g) City-wise Distribution')
    pdf.savefig(fig7,bbox_inches = 'tight')
    plt.close()
    
    fig8 = plt.figure()
    df['College name'].value_counts().plot(kind='barh',figsize=(12, 8),color='Gray')
    plt.xlabel('Number of Students')
    plt.ylabel('College Name')
    plt.title('Q2(g) College-wise Distribution')
    pdf.savefig(fig8,bbox_inches = 'tight')
    plt.close()
    
    fig9=plt.figure(figsize=(15,8))
    plt.title('Q2(h) Plot the relationship between the CGPA and the target variable.')
    sns.violinplot(x="Label", y="CGPA/ percentage", data=df)
    pdf.savefig(fig9,bbox_inches = 'tight')
    plt.close()
    
    labels = list(set(df["Areas of interest"]))
    df2i_eligible=df[["Areas of interest","Label"]].query("Label=='eligible'").groupby("Areas of interest")
    eligible=df2i_eligible['Label'].value_counts().tolist()
    df2i_ineligible=df[["Areas of interest","Label"]].query("Label=='ineligible'").groupby("Areas of interest")
    ineligible=df2i_ineligible['Label'].value_counts().tolist()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig10, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(x - width/2, eligible, width, label='Eligible')
    rects2 = ax.bar(x + width/2, ineligible, width, label='Ineligible')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Students')
    ax.set_xlabel('Areas of Interest')
    ax.set_title('Q2(i) Plot the relationship between the Area of Interest and the target variable.')
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=45, ha='right')
    ax.legend()
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig10.tight_layout()
    pdf.savefig(fig10,bbox_inches = 'tight')
    plt.close()
    
    df2j_eligible=df[['Which-year are you studying in?',"Major/Area of Study",'Label']].query("Label=='eligible'")
    df2j_ineligible=df[['Which-year are you studying in?',"Major/Area of Study",'Label']].query("Label=='ineligible'")
    #eligible
    eligible_ce=df2j_eligible[df2j_eligible['Major/Area of Study'] == 'Computer Engineering'].groupby('Which-year are you studying in?')['Label'].value_counts().tolist()
    eligible_ee=df2j_eligible[df2j_eligible['Major/Area of Study'] == 'Electrical Engineering'].groupby('Which-year are you studying in?')['Label'].value_counts().tolist()
    eligible_extc=df2j_eligible[df2j_eligible['Major/Area of Study'] == 'Electronics and Telecommunication'].groupby('Which-year are you studying in?')['Label'].value_counts().tolist()
    #ineligible
    ineligible_ce=df2j_ineligible[df2j_ineligible['Major/Area of Study'] == 'Computer Engineering'].groupby('Which-year are you studying in?')['Label'].value_counts().tolist()
    ineligible_ee=df2j_ineligible[df2j_ineligible['Major/Area of Study'] == 'Electrical Engineering'].groupby('Which-year are you studying in?')['Label'].value_counts().tolist()
    ineligible_extc=df2j_ineligible[df2j_ineligible['Major/Area of Study'] == 'Electronics and Telecommunication'].groupby('Which-year are you studying in?')['Label'].value_counts().tolist()
    if(len(eligible_ce)==3):
        eligible_ce.insert(0,0)
    if(len(eligible_ee)==3):
        eligible_ee.insert(0,0)
    if(len(eligible_extc)==3):
        eligible_extc.insert(0,0)
    years =['First-year', 'Second-year', 'Third-year', 'Fourth-year']
    with sns.axes_style("white"):
        sns.set_style("ticks")
        sns.set_context("talk")
        # plot details
        bar_width = 0.35
        epsilon = .015
        line_width = 1
        opacity = 0.7
        pos_bar_positions = np.arange(len(eligible_ce))
        neg_bar_positions = pos_bar_positions + bar_width
        # make bar plots
        fig11 = plt.figure(figsize=(15,8))
        eligible_ce_bar = plt.bar(pos_bar_positions, eligible_ce, bar_width,
                                  color='#ED0020',
                                  label='Eligible CSE')
        eligible_ee_bar = plt.bar(pos_bar_positions, eligible_ee, bar_width-epsilon,
                                  bottom=eligible_ce,
                                  alpha=opacity,
                                  color='white',
                                  edgecolor='#ED0020',
                                  linewidth=line_width,
                                  hatch='//',
                                  label='Eligible EE')
        eligible_extc_bar = plt.bar(pos_bar_positions, eligible_extc, bar_width-epsilon,
                                   bottom=np.add(eligible_ee,eligible_ce),
                                   alpha=opacity,
                                   color='white',
                                   edgecolor='#ED0020',
                                   linewidth=line_width,
                                   hatch='0',
                                   label='Eligible EXTC')
        ineligible_ce_bar = plt.bar(neg_bar_positions, ineligible_ce, bar_width,
                                  color='#0000DD',
                                  label='Ineligible CSE')
        ineligible_ee_bar = plt.bar(neg_bar_positions, ineligible_ee, bar_width-epsilon,
                                  bottom=ineligible_ce,
                                  color="white",
                                  hatch='//',
                                  edgecolor='#0000DD',
                                  ecolor="#0000DD",
                                  linewidth=line_width,
                                  label='Ineligible EE')
        ineligible_extc_bar = plt.bar(neg_bar_positions, ineligible_extc, bar_width-epsilon,
                                   bottom=np.add(ineligible_ee,ineligible_ce),
                                   color="white",
                                   hatch='0',
                                   edgecolor='#0000DD',
                                   ecolor="#0000DD",
                                   linewidth=line_width,
                                   label='Ineligible Extc')
        plt.xticks((neg_bar_positions + pos_bar_positions)//2, years, rotation=45)
        plt.ylabel('No. of Students')
        plt.title('Q2(h) Plot the relationship between the year of study, major, and the target variable.',pad=50)
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        # fig11=plt.figure(figsize=(15,8))
        sns.despine()
        pdf.savefig(fig11,bbox_inches = 'tight')
        plt.close()
