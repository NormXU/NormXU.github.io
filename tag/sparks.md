---
layout: tag
title: "Tag: Sparks"
permalink: /t/sparks
---
This page is a repository for collecting my inspirational sparks, which I’m unable to experiment due to limited access to data and computational resources. Hope that my ideas can inspire you. If you find some of them interesting and would like to explore further, I would greatly appreciate it if you allow me to join you.
<ul class="post-list">
  {%- for post in site.tags["Sparks"] -%}
    <li>
      {%- assign date_format = site.minima.date_format | default: "%Y-%m-%d" -%}
      <span class="post-meta">
        {{ post.date | date: date_format }}
      </span>
      <a class="post-link" href="{{ post.url | relative_url }}">
          {{ post.title | escape }}
      </a>
    </li>
  {%- endfor -%}
</ul>