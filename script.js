
var valid = "https://training.pada-x.com/wehelp/";
var input;
var entry_count=0;
var entry_limit=3;
var left_count=3;
var out_limit=false;



while(valid!=input && !out_limit){
    entry_count++;
    if(entry_count<=entry_limit){
        input=prompt("please type in we help url, you can try: "+left_count+" times");
        left_count--;
    }
    else{
        out_limit=true;
    }
}

if (out_limit){
    alert("you can not review the content. close this tab.");
    window.open('', '_self', ''); //bug fix    
    window.close();
    open(location, '_self').close();
    
}
else{
    alert("welcome!");
    out_limit=true;
    
}



