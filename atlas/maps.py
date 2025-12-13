def get_box_div_style(left_or_right: str):
    return f"""
"
    position: fixed; 
    bottom: 40px;
    {left_or_right}: 40px;
    width: 260px;
    z-index:9999;
    background-color: white;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 0 8px rgba(0,0,0,0.2);
    font-size:14px;
"
"""

def get_legend_div_style():
    return get_box_div_style('right')

def get_help_div_style():
    return get_box_div_style('left')

def get_help_footer():
    return """
<br>This map is scrollable and zoomable.<br>
<p><br>
<a href="https://github.com/nbirnel/atlas-of-eugene">Source Code for this Map</a>
</p>    
"""

def get_title_html(title: str):
    return f'<h1 style="position:absolute;z-index:100000;left:20vw" >{title}</h1>'


